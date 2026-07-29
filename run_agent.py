# run_agent.py
import json
import subprocess
from datetime import datetime
from openai import OpenAI

# =========================================
# 步骤1：构建物理工具与沙盒执行器
# =========================================
def execute_python_code(code_str: str) -> str:
    """沙盒内执行动态生成的Python代码。"""
    try:
        print(f"\n[系统沙盒] 正在执行生成的代码:\n{code_str}\n")
        # 使用独立的进程执行以确保主控制循环的安全[cite: 1]
        result = subprocess.run(
            ['python3', '-c', code_str],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"执行失败。错误输出: {result.stderr.strip()}"
    except Exception as e:
        return f"系统异常: {str(e)}"

def get_system_environment() -> str:
    """获取当前宿主机环境的上下文信息。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"当前系统时间: {now}。系统类型: AMD ROCm 部署节点。"

# 定义可供模型识别的标准工具签名 Schema[cite: 1]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_python_code",
            "description": "执行传入的Python代码字符串。当需要进行数学计算、分析数据或自动化文件处理时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code_str": {
                        "type": "string",
                        "description": "合法的Python 3 脚本字符串。例如 'print(sum(range(100)))'"
                    }
                },
                "required": ["code_str"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_environment",
            "description": "查询当前服务器的时间和基础环境信息。"
        }
    }
]

# =========================================
# 步骤2：ReAct 代理主推理控制流
# =========================================
def run_agent_loop(user_query: str):
    # 初始化指向本地AMD节点上vLLM的客户端[cite: 1]
    client = OpenAI(
        api_key="sk-local-amd",
        base_url="http://localhost:8000/v1"
    )

    # 确立代理系统定位[cite: 1]
    messages = [
        {"role": "system", "content": "你是一个运行在AMD Radeon GPU上的高级自动推理代码代理。你能够拆解复杂任务，并通过编写代码来获取所需信息或改变系统状态。请在必要时果断调用工具。"},
        {"role": "user", "content": user_query}
    ]

    print(f"\n======================================")
    print(f"[任务下达]: {user_query}")
    print(f"======================================\n")

    max_iterations = 6
    for step in range(max_iterations):
        print(f"[{step+1}/{max_iterations}] 代理正在进行思考规划...")
        
        # 请求底层大模型进行意图推理，使用低温度增强结构化输出的稳定性[cite: 1]
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=messages,
            tools=TOOLS,
            temperature=0.1
        )
        
        response_message = response.choices[0].message
        
        # 判断模型是否生成了工具调用指令[cite: 1]
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                try:
                    kwargs = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    print(f"[解析警告] 模型生成了格式错误的JSON参数: {tool_call.function.arguments}")
                    kwargs = {}

                print(f"[行动决策] 调度函数 -> {func_name}()")
                
                # 物理工具路由与执行[cite: 1]
                if func_name == "execute_python_code":
                    obs = execute_python_code(kwargs.get("code_str", ""))
                elif func_name == "get_system_environment":
                    obs = get_system_environment()
                else:
                    obs = f"Error: 找不到请求的函数 {func_name}。"

                print(f"[工具观测结果] -> {obs}")
                
                # 记忆管理：存入上下文供下一步推理使用[cite: 1]
                messages.append(response_message)
                messages.append({
                    "role": "tool",
                    "name": func_name,
                    "content": obs,
                    "tool_call_id": tool_call.id
                })
        else:
            # 模型没有输出工具调用，判定为已经得出最终结论[cite: 1]
            print("\n======================================")
            print("[代理最终回复]:")
            print(response_message.content)
            print("======================================\n")
            break

if __name__ == "__main__":
    # 综合测试场景：融合时间查询、代码生成与计算执行[cite: 1]
    test_query = "获取当前的系统时间。然后写一段Python代码，计算出第100个斐波那契数，并告诉我答案。"
    run_agent_loop(test_query)