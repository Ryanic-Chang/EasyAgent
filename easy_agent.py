"""
EasyAgent - AMD Radeon GPU 智能代理系统
黑客松 Track 2: Agentic AI

核心特性：
1. ReAct推理循环（思考→行动→观察→重复）
2. Qwen2.5原生Function Calling（chat template）
3. 8个专业工具（代码执行、文件操作、RAG检索、GPU监控等）
4. 双层记忆管理（短期+长期）
5. AMD ROCm优化推理 (~18 tokens/s)
6. Gradio Web UI
"""
import os
import sys
import json
import subprocess
import torch
import datetime
import re
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from transformers import TextStreamer, StoppingCriteria


# =========================================
# 流式输出 + 文本捕获
# =========================================
class CaptureStreamer(TextStreamer):
    """流式打印token到stdout，同时捕获完整生成文本"""
    def __init__(self, tokenizer, skip_prompt=True):
        super().__init__(tokenizer, skip_prompt=skip_prompt)
        self._captured = []
        self.repetition_detected = False

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self._captured.append(text)
        # 检测重复循环：最近512字符中是否有长重复段
        full = ''.join(self._captured)
        if len(full) > 200:
            tail = full[-512:]
            for pat_len in (32, 16, 8):
                pat = tail[-pat_len:]
                if tail[-pat_len*4:-pat_len] == pat * 3:
                    self.repetition_detected = True
                    break
        sys.stdout.write(text)
        sys.stdout.flush()
        if stream_end:
            sys.stdout.write('\n')
            sys.stdout.flush()

    @property
    def captured_text(self) -> str:
        return ''.join(self._captured)


class RepetitionStopping(StoppingCriteria):
    """当检测到重复循环时提前停止生成"""
    def __init__(self, tokenizer, window=256, min_pat=8):
        self.tokenizer = tokenizer
        self.window = window
        self.min_pat = min_pat

    def __call__(self, input_ids, scores, **kwargs):
        # 只检查最新生成的token
        gen_ids = input_ids[0][input_ids.shape[1]:] if input_ids.shape[1] > 0 else input_ids[0]
        if len(gen_ids) < self.window:
            return False
        recent = gen_ids[-self.window:].tolist()
        # 检查尾部是否有重复的token序列
        for pat_len in (self.min_pat, self.min_pat * 2):
            if len(recent) < pat_len * 4:
                continue
            pat = recent[-pat_len:]
            chunk = recent[-pat_len*3:-pat_len]
            if chunk == pat * 2:
                return True
        return False


# =========================================
# 记忆管理
# =========================================
@dataclass
class MemoryItem:
    content: str
    timestamp: str
    importance: int = 1


class MemoryManager:
    """双层记忆：短期记忆自动沉淀为长期记忆"""
    def __init__(self):
        self.short_term: List[MemoryItem] = []
        self.long_term: List[MemoryItem] = []
        self.max_short = 10
        self.max_long = 50

    def add(self, content: str, importance: int = 1, term: str = "short"):
        item = MemoryItem(
            content=content,
            timestamp=datetime.datetime.now().strftime("%H:%M:%S"),
            importance=importance
        )
        target = self.short_term if term == "short" else self.long_term
        target.append(item)
        if len(target) > (self.max_short if term == "short" else self.max_long):
            if term == "short":
                old = target.pop(0)
                self.long_term.append(old)
            else:
                target.sort(key=lambda x: x.importance)
                target.pop(0)

    def get_context(self) -> str:
        parts = []
        if self.short_term:
            parts.append("[近期记忆]")
            for item in self.short_term[-5:]:
                parts.append(f"  [{item.timestamp}] {item.content}")
        if self.long_term:
            parts.append("[重要记忆]")
            for item in sorted(self.long_term, key=lambda x: x.importance, reverse=True)[:3]:
                parts.append(f"  - {item.content[:80]}")
        return "\n".join(parts)

    def save(self, path="memory.json"):
        data = {
            "short": [vars(i) for i in self.short_term],
            "long": [vars(i) for i in self.long_term]
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path="memory.json"):
        if not os.path.exists(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.short_term = [MemoryItem(**i) for i in data.get("short", [])]
        self.long_term = [MemoryItem(**i) for i in data.get("long", [])]


# =========================================
# 工具系统
# =========================================
class ToolSystem:
    """工具注册与执行"""

    def __init__(self):
        self.tools: Dict[str, Dict] = {}
        self._register_all()

    def _register_all(self):
        self._reg("execute_code",
            "执行Python代码字符串，返回stdout输出。用于数学计算、数据处理、自动化任务。",
            {"type": "object", "properties": {"code": {"type": "string", "description": "Python代码"}}, "required": ["code"]},
            self._exec_code)

        self._reg("system_info",
            "获取系统时间、GPU型号、显存使用、ROCm版本等信息",
            {"type": "object", "properties": {}, "required": []},
            self._sys_info)

        self._reg("read_file",
            "读取文件内容(最大2000字符)",
            {"type": "object", "properties": {"path": {"type": "string", "description": "文件路径"}}, "required": ["path"]},
            self._read_file)

        self._reg("write_file",
            "写入内容到文件",
            {"type": "object", "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "content": {"type": "string", "description": "文件内容"}
            }, "required": ["path", "content"]},
            self._write_file)

        self._reg("list_dir",
            "列出目录内容",
            {"type": "object", "properties": {"path": {"type": "string", "description": "目录路径，默认'.'"}}, "required": []},
            self._list_dir)

        self._reg("search_knowledge",
            "检索本地知识库(AMD GPU/ROCm/vLLM/Qwen/RAG/Agent)",
            {"type": "object", "properties": {"query": {"type": "string", "description": "搜索关键词"}}, "required": ["query"]},
            self._search_kb)

        self._reg("gpu_monitor",
            "获取GPU使用率、温度、显存占用详情",
            {"type": "object", "properties": {}, "required": []},
            self._gpu_monitor)

        self._reg("calculate",
            "执行数学表达式计算",
            {"type": "object", "properties": {"expr": {"type": "string", "description": "数学表达式"}}, "required": ["expr"]},
            self._calculate)

    def _reg(self, name, desc, params_schema, func):
        self.tools[name] = {
            "description": desc,
            "schema": params_schema,
            "func": func,
            "openai_schema": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": params_schema
                }
            }
        }

    def _exec_code(self, code: str) -> str:
        try:
            r = subprocess.run(['python3', '-c', code],
                             capture_output=True, text=True, timeout=15)
            out = r.stdout.strip()
            err = r.stderr.strip()
            if r.returncode == 0:
                return out if out else "执行成功（无输出）"
            return f"错误: {err[:300]}"
        except subprocess.TimeoutExpired:
            return "执行超时(>15秒)"
        except Exception as e:
            return f"异常: {e}"

    def _sys_info(self) -> str:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu = f"{props.name} (gfx1100 RDNA3, 96CU)"
            vram_total = f"{props.total_memory/1024**3:.1f}GB"
            vram_used = f"{torch.cuda.memory_allocated()/1024**3:.2f}GB"
            vram_free = f"{(props.total_memory - torch.cuda.memory_allocated())/1024**3:.2f}GB"
        else:
            gpu = "无"
            vram_total = vram_used = vram_free = "N/A"
        return (f"时间: {now}\n"
                f"GPU: {gpu}\n"
                f"显存: {vram_used}/{vram_total} (空闲{vram_free})\n"
                f"ROCm: 6.2 | PyTorch: {torch.__version__}")

    def _read_file(self, path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                c = f.read()
            return c[:2000] + ("..." if len(c) > 2000 else "")
        except Exception as e:
            return f"读取失败: {e}"

    def _write_file(self, path: str, content: str) -> str:
        try:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"写入{len(content)}字节到 {path}"
        except Exception as e:
            return f"写入失败: {e}"

    def _list_dir(self, path: str = ".") -> str:
        try:
            entries = os.listdir(path)
            result = []
            for e in sorted(entries)[:30]:
                full = os.path.join(path, e)
                t = "[DIR]" if os.path.isdir(full) else "[FILE]"
                result.append(f"  {t} {e}")
            return f"目录 {path}:\n" + "\n".join(result)
        except Exception as e:
            return f"列出失败: {e}"

    def _search_kb(self, query: str) -> str:
        kb = {
            "amd": "AMD Radeon RX 7900 (gfx1100) RDNA3架构，96CU，48GB显存，适合AI推理",
            "rocm": "ROCm 6.2是AMD GPU计算平台，支持PyTorch/TensorFlow，类似CUDA",
            "vllm": "vLLM: 高性能LLM推理引擎，核心技术PagedAttention+连续批处理",
            "qwen": "Qwen2.5-7B-Instruct: 通义千问指令微调7B模型，支持function calling",
            "react": "ReAct: Reasoning+Acting范式，思考->行动->观察循环",
            "rag": "RAG: 检索增强生成，先检索相关知识再生成回答",
            "agent": "AI Agent: 能自主推理、规划、使用工具的智能体",
            "rdna": "RDNA 3: AMD最新GPU架构，支持AI加速、光线追踪、AV1编解码",
            "gpu": "48GB VRAM，适合运行7B-13B参数模型，约18 tokens/s推理速度"
        }
        results = []
        q = query.lower()
        for key, val in kb.items():
            if key in q or any(w in q for w in key.split()):
                results.append(f"[{key.upper()}] {val}")
        return "\n".join(results) if results else "未找到相关知识，尝试: amd/rocm/vllm/qwen/react/rag/agent"

    def _gpu_monitor(self) -> str:
        try:
            r = subprocess.run(['rocm-smi', '--showuse', '--showmemuse', '--showtemp'],
                             capture_output=True, text=True, timeout=5)
            lines = r.stdout.split('\n')
            info = [l.strip() for l in lines if any(k in l for k in ['GPU', 'Temp', 'Power', '%'])]
            return '\n'.join(info[:8]) if info else r.stdout[:500]
        except Exception as e:
            return f"监控异常: {e}"

    def _calculate(self, expr: str) -> str:
        try:
            allowed = set("0123456789+-*/().% ")
            if not all(c in allowed or c.isalpha() for c in expr):
                return "包含非法字符"
            result = eval(expr, {"__builtins__": {}}, {
                "sum": sum, "range": range, "abs": abs, "round": round,
                "min": min, "max": max, "pow": pow, "int": int, "float": float
            })
            return f"{expr} = {result}"
        except Exception as e:
            return f"计算错误: {e}"

    def get_openai_schemas(self) -> List[Dict]:
        return [t["openai_schema"] for t in self.tools.values()]

    def execute(self, name: str, args: dict) -> str:
        if name not in self.tools:
            return f"工具'{name}'不存在。可用: {', '.join(self.tools.keys())}"
        try:
            return self.tools[name]["func"](**args)
        except TypeError as e:
            return f"参数错误: {e}\n需要参数: {list(self.tools[name]['schema'].get('properties', {}).keys())}"
        except Exception as e:
            return f"执行失败: {e}"


# =========================================
# EasyAgent核心
# =========================================
class EasyAgent:
    """ReAct智能代理 - 使用Qwen2.5原生Function Calling"""

    def __init__(self, model_path: str):
        self._banner()
        print("[1/3] 加载Qwen2.5-7B模型...")
        t0 = datetime.datetime.now()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float16, device_map="auto"
        )
        elapsed = (datetime.datetime.now() - t0).total_seconds()
        print(f"   OK: 模型就绪 [{elapsed:.1f}s, device={self.device}]")

        self.tools = ToolSystem()
        self.memory = MemoryManager()
        try:
            self.memory.load()
        except Exception:
            pass
        print(f"   OK: {len(self.tools.tools)}个工具已注册")
        print(f"   OK: 记忆系统就绪\n")

    def _banner(self):
        print()
        print("=" * 65)
        print("  EasyAgent - AMD Radeon GPU 智能代理")
        print("  Hackathon Track 2: Agentic AI")
        print("  本地推理: Qwen2.5-7B on AMD gfx1100 (48GB VRAM)")
        print("=" * 65)

    def _build_system_prompt(self) -> str:
        mem_ctx = self.memory.get_context()
        prompt = """你是EasyAgent，一个运行在AMD Radeon GPU上的智能AI代理。

你的核心能力：
- 分析复杂任务，拆解为可执行步骤
- 在需要时调用工具执行操作
- 管理短期和长期记忆
- 生成和执行Python代码

工作原则：
- 需要信息时主动调用工具，不要臆测
- 每次只调用一个工具
- 根据工具返回结果调整策略
- 最终给出简洁、有条理的回答

工具选择指南：
- 生成HTML/CSS/JS/Python等完整代码文件时，必须使用 write_file 工具直接写入文件，不要用 execute_code 包裹代码
- write_file 的参数 content 直接写原始代码内容，不要包裹在 Python 字符串里
- 如果代码较长，分多次 write_file 写入同一文件（追加模式不可用，请一次性写完整）"""
        if mem_ctx:
            prompt += f"\n\n你的记忆:\n{mem_ctx}"
        return prompt

    def run(self, query: str, max_steps: int = 5, callback: Optional[Callable] = None) -> str:
        """运行Agent ReAct循环"""
        self.memory.add(f"任务: {query[:50]}", importance=2)
        if callback:
            callback({"type": "task", "content": query})

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": query}
        ]

        tool_schemas = self.tools.get_openai_schemas()

        for step in range(max_steps):
            if callback:
                callback({"type": "step", "step": step + 1, "max": max_steps})

            # 使用Qwen2.5原生chat template + function calling
            prompt = self.tokenizer.apply_chat_template(
                messages, tools=tool_schemas,
                tokenize=False, add_generation_prompt=True
            )

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            streamer = CaptureStreamer(self.tokenizer, skip_prompt=True)
            stopping = RepetitionStopping(self.tokenizer)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=16384,
                    temperature=0.6,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.08,
                    no_repeat_ngram_size=6,
                    pad_token_id=self.tokenizer.eos_token_id,
                    streamer=streamer,
                    stopping_criteria=[stopping]
                )

            # 使用流式捕获的文本（已实时打印）
            response_text = streamer.captured_text.strip()
            if streamer.repetition_detected:
                print("\n[检测到重复循环，已提前截断]")
                response_text = self._clean_response(response_text)
            
            # 截断：工具调用后可能有重复生成
            ot = chr(60) + "tool_call" + chr(62)
            ct = chr(60) + "/tool_call" + chr(62)
            if ot in response_text:
                end_idx = response_text.find(ct)
                if end_idx != -1:
                    response_text = response_text[:end_idx + len(ct)]

            # 解析function calling
            tool_calls = self._parse_function_calls(response_text)

            if tool_calls:
                # 添加assistant消息
                messages.append({"role": "assistant", "content": response_text})

                # 执行工具调用
                for tc in tool_calls:
                    name = tc.get("name", "")
                    args = tc.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}

                    if callback:
                        callback({"type": "tool_call", "name": name, "args": args})

                    result = self.tools.execute(name, args)

                    if callback:
                        callback({"type": "tool_result", "name": name, "result": result})

                    self.memory.add(f"执行{name}: {result[:50]}", importance=1)

                    messages.append({
                        "role": "tool",
                        "content": result,
                        "name": name
                    })
            else:
                # 最终回答
                answer = self._clean_response(response_text)
                if callback:
                    callback({"type": "answer", "content": answer})
                self.memory.add(f"完成: {query[:30]}", importance=3, term="long")
                self.memory.save()
                return answer

        return "达到最大步数限制"

    def _parse_function_calls(self, text: str) -> List[Dict]:
        calls = []
        ot = chr(60) + "tool_call" + chr(62)
        ct = chr(60) + "/tool_call" + chr(62)
        while ot in text:
            start = text.index(ot) + len(ot)
            end = text.find(ct, start)
            if end == -1: break
            chunk = text[start:end].strip()
            try:
                calls.append(json.loads(chunk))
            except json.JSONDecodeError:
                pass
            text = text[end + len(ct):]
        return calls

    def _clean_response(self, text: str) -> str:
        ot = chr(60) + "tool_call" + chr(62)
        ct = chr(60) + "/tool_call" + chr(62)
        while ot in text:
            start = text.index(ot)
            end = text.find(ct, start)
            if end == -1: break
            text = text[:start] + text[end + len(ct):]
        # 去除重复字符序列（如!!!... 或 啊啊啊...）
        import re as _re
        text = _re.sub(r"(.)\1{4,}", lambda m: m.group(1) * 2, text)
        # 去除尾部连续重复标点
        text = _re.sub(r"([!?.])\1{2,}$", r"\1", text)
        # 去除尾部未闭合的代码块（截断标志）
        if text.rstrip().endswith('{'):
            text = _re.sub(r'\{[^{}]*$', '', text.rstrip()).rstrip()
        # 去除重复的行（连续相同的行只保留一次）
        lines = text.split('\n')
        deduped = []
        for line in lines:
            if deduped and deduped[-1] == line and line.strip() in ('', '!!!', '...', '---'):
                continue
            deduped.append(line)
        text = '\n'.join(deduped)
        return text.strip()

# =========================================
# Gradio Web UI
# =========================================
def create_web_ui(agent):
    import gradio as gr

    def chat(message, history):
        if not message.strip():
            yield 'Please enter a task'
            return

        step_log = []

        def cb(info):
            t = info.get('type', '')
            if t == 'task':
                step_log.append(f'Task: {info["content"]}')
            elif t == 'step':
                step_log.append(f'Step {info["step"]}/{info["max"]}')
            elif t == 'tool_call':
                step_log.append(f'Tool: {info["name"]}')
            elif t == 'tool_result':
                step_log.append(f'Result: {info["result"][:150]}')
            elif t == 'answer':
                step_log.append(f'Answer: {info["content"]}')

        agent.run(message, max_steps=4, callback=cb)
        yield '\n'.join(step_log)

    with gr.Blocks(title='EasyAgent - AMD GPU Agent') as demo:
        gr.Markdown('# EasyAgent - AMD Radeon GPU Agent\nTrack 2: Agentic AI Hackathon')

        with gr.Row():
            with gr.Column(scale=3):
                gr.ChatInterface(fn=chat, title='Chat with EasyAgent',
                    examples=['Get system info and GPU status',
                              'Calculate the sum of first 100 primes',
                              'Search knowledge about AMD GPU and ROCm'])
            with gr.Column(scale=1):
                gr.Markdown('### System Status')
                btn = gr.Button('Refresh', variant='primary')
                status = gr.Markdown()
                btn.click(lambda: agent.tools.execute('system_info', {}) + '\n\n' + agent.tools.execute('gpu_monitor', {}), outputs=status)

                gr.Markdown('### Available Tools')
                tool_list = '\n'.join([f'- {n}: {i["description"][:50]}' for n, i in agent.tools.tools.items()])
                gr.Markdown(tool_list)

        gr.Markdown('---\nBuilt for AMD Hackathon 2026')

    return demo


# =========================================
# Main Entry
# =========================================
if __name__ == '__main__':
    import sys
    MODEL = '/workspace/EasyAgent/models/Qwen2.5-7B-Instruct'
    agent = EasyAgent(MODEL)

    if '--web' in sys.argv:
        demo = create_web_ui(agent)
        demo.launch(server_name='127.0.0.1', server_port=7860, share=True)
    elif '--demo' in sys.argv:
        demos = ['Get current system time and GPU status',
                 'Calculate the sum of first 100 prime numbers and save to a file',
                 'Search knowledge about AMD GPU']
        for q in demos:
            agent.run(q, max_steps=3)
            print('\n' + '-' * 50)
    else:
        print('\nEnter task (type quit to exit):')
        while True:
            try:
                query = input('\n> ').strip()
                if query.lower() in ('quit', 'exit', 'q'):
                    break
                if query:
                    print()
                    try:
                        result = agent.run(query)
                        print()
                        print('─' * 55)
                        print(f'  {result}')
                        print('─' * 55)
                    except Exception as e:
                        print(f'\n  [ERROR] {e}')
                        print('─' * 55)
            except (KeyboardInterrupt, EOFError):
                break
        print('\nBye!')
        agent.memory.save()
