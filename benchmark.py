"""
EasyAgent 性能基准测试
AMD Radeon GPU ROCm 优化评估
"""
import os
import json
import time
import torch
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_benchmark(model_path="/workspace/EasyAgent/models/Qwen2.5-7B-Instruct"):
    """运行完整性能基准测试"""
    print("=" * 70)
    print("  EasyAgent 性能基准测试")
    print("  AMD Radeon GPU + ROCm 6.2")
    print("=" * 70)
    print()

    # 1. 系统信息
    print("[1/6] 系统信息")
    print(f"  时间: {datetime.datetime.now()}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA(ROCm)可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}")
        print(f"  显存总量: {props.total_memory / 1024**3:.2f} GB")
        print(f"  计算能力: gfx{props.major}{props.minor}")
    print()

    # 2. 模型加载时间
    print("[2/6] 模型加载测试")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    t1 = time.time()
    print(f"  Tokenizer加载: {t1-t0:.2f}s")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="auto"
    )
    t2 = time.time()
    print(f"  模型加载: {t2-t1:.2f}s")
    print(f"  总加载时间: {t2-t0:.2f}s")
    print(f"  显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print()

    # 3. 推理速度测试
    print("[3/6] 推理速度测试")
    test_cases = [
        ("短文本", "Hello, how are you?", 20),
        ("中等文本", "Explain quantum computing in simple terms.", 50),
        ("长文本生成", "Write a detailed paragraph about AMD Radeon GPUs and their features.", 100),
        ("代码生成", "Write a Python function to calculate fibonacci numbers.", 80),
    ]

    results = []
    for name, prompt, max_tokens in test_cases:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # 预热
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)

        # 正式测试
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id
            )
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        speed = tokens / elapsed

        results.append({
            "name": name,
            "tokens": tokens,
            "time": elapsed,
            "speed": speed
        })
        print(f"  {name}: {tokens} tokens, {elapsed:.2f}s, {speed:.1f} tokens/s")

    avg_speed = sum(r["speed"] for r in results) / len(results)
    print(f"\n  平均速度: {avg_speed:.1f} tokens/s")
    print()

    # 4. 工具调用测试
    print("[4/6] Agent工具调用测试")
    from easy_agent import EasyAgent
    agent = EasyAgent.__new__(EasyAgent)
    agent.tokenizer = tokenizer
    agent.model = model
    agent.device = "cuda"
    from easy_agent import ToolSystem, MemoryManager
    agent.tools = ToolSystem()
    agent.memory = MemoryManager()

    test_queries = [
        "What time is it?",
        "Tell me about AMD GPU",
        "Calculate 2^10 + 3^5",
    ]

    for q in test_queries:
        t0 = time.time()
        r = agent.run(q, max_steps=2)
        elapsed = time.time() - t0
        print(f"  任务: {q[:30]}...")
        print(f"  耗时: {elapsed:.2f}s")
        print(f"  结果: {r[:80]}...")
        print()

    # 5. 显存效率
    print("[5/6] 显存效率")
    print(f"  模型占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"  缓存占用: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    print(f"  总显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    efficiency = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
    print(f"  显存利用率: {efficiency:.1f}%")
    print()

    # 6. 汇总
    print("[6/6] 汇总报告")
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "vram_used_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
        "model_load_time_s": round(t2 - t0, 2),
        "avg_inference_speed_tokens_per_s": round(avg_speed, 1),
        "rocm_version": "6.2",
        "pytorch_version": torch.__version__,
        "tests": results
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    # 保存报告
    report_path = "/workspace/EasyAgent/benchmark_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {report_path}")

    return report


if __name__ == "__main__":
    run_benchmark()
