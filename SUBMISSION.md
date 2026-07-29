# 黑客松提交材料

## Track 2: Agentic AI - EasyAgent

### 项目规范文档

#### 应用场景

**EasyAgent** 是一个运行在 AMD Radeon GPU 上的智能 AI 代理系统，适用于以下场景：

1. **个人生产力助手** - 自动化日常任务（计算、文件操作、信息查询）
2. **开发者工具** - 代码生成、执行、调试的 AI 辅助
3. **本地知识助手** - 基于 RAG 的知识检索和问答
4. **系统监控代理** - GPU 资源监控和报告
5. **工作流自动化** - 多步骤任务的自动规划和执行

#### 代理架构图

```
┌─────────────────────────────────────────────────────────┐
│                      EasyAgent                           │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │              ReAct 推理循环                        │  │
│  │         思考 → 行动 → 观察 → 重复                  │  │
│  │    (最多5步迭代，直到完成任务或达到限制)              │  │
│  └──────────────────────────────────────────────────┘  │
│                          │                               │
│        ┌─────────────────┼─────────────────┐            │
│        ▼                 ▼                 ▼            │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐       │
│  │ 记忆管理  │     │ 工具系统  │     │ 模型推理  │       │
│  │          │     │          │     │          │       │
│  │ • 短期记忆│     │ • 代码执行│     │ Qwen2.5  │       │
│  │ • 长期记忆│     │ • 文件操作│     │ 7B参数    │       │
│  │ • 自动沉淀│     │ • 知识检索│     │ Function │       │
│  │ • 持久化  │     │ • GPU监控 │     │ Calling  │       │
│  └──────────┘     │ • 数学计算│     │          │       │
│                   └──────────┘     └──────────┘       │
│                                         │               │
│                                         ▼               │
│                              ┌──────────────────┐      │
│                              │  AMD Radeon GPU   │      │
│                              │  gfx1100 RDNA3    │      │
│                              │  48GB VRAM        │      │
│                              │  ROCm 6.2         │      │
│                              └──────────────────┘      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### 核心功能介绍

**1. ReAct 推理循环**
- 基于 Reasoning + Acting 范式
- 智能分解复杂任务为可执行步骤
- 每步调用工具获取信息或执行操作
- 根据观察结果动态调整策略

**2. 多工具调用系统**
- **execute_code**: 执行 Python 代码，返回输出
- **system_info**: 获取系统时间、GPU、显存信息
- **read_file / write_file**: 文件读写操作
- **list_dir**: 目录浏览
- **search_knowledge**: 本地知识库检索（RAG）
- **gpu_monitor**: GPU 使用率、温度监控
- **calculate**: 安全的数学表达式计算

**3. 双层记忆管理**
- **短期记忆**: 记录近期操作和结果（最多10条）
- **长期记忆**: 按重要性保存关键信息（最多50条）
- **自动沉淀**: 短期记忆自动转为长期记忆
- **持久化**: 记忆保存到 JSON 文件，跨会话保留

**4. Qwen2.5 原生 Function Calling**
- 使用 Qwen2.5 的 chat template
- 可靠的 JSON 格式工具调用
- 支持多参数传递
- 错误处理和重试机制

**5. Web UI 界面**
- 基于 Gradio 的交互式界面
- 实时显示推理过程
- 工具调用可视化
- 系统状态监控

#### 模型介绍及本地部署计划

**模型选择：Qwen2.5-7B-Instruct**

| 属性 | 值 |
|------|-----|
| 参数量 | 7B |
| 架构 | Transformer Decoder |
| 训练数据 | 多语言语料 |
| 特色 | 指令微调、Function Calling |
| 量化 | FP16 |
| 显存需求 | 14.19 GB |

**本地部署方案：**
1. **模型托管**: 本地文件系统 (`/workspace/EasyAgent/models/`)
2. **推理框架**: HuggingFace Transformers + PyTorch
3. **GPU加速**: ROCm 6.2 + FP16 推理
4. **API服务**: 可选 vLLM 或 Gradio Server
5. **启动时间**: 5秒（模型加载）

**部署优势：**
- ✅ 完全本地化，数据隐私安全
- ✅ 无网络依赖，离线可用
- ✅ 低延迟（本地 GPU 推理）
- ✅ 成本可控（无 API 调用费用）

#### AMD Radeon GPU 推理速度优化描述

**硬件配置：**
- GPU: AMD Radeon Graphics (gfx1100, RDNA 3)
- 计算单元: 96 CU
- 显存: 48 GB VRAM
- ROCm: 6.2

**优化措施：**

1. **FP16 半精度推理**
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       model_path, dtype=torch.float16, device_map="auto"
   )
   ```
   - 减少显存占用 50%
   - 提升计算吞吐量

2. **PyTorch ROCm 优化**
   ```bash
   export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
   export HIP_FORCE_DEV_KERNARG=1
   export HSA_OVERRIDE_GFX_VERSION="11.0.0"
   ```
   - 优化 GPU 内存分配
   - 强制使用设备内核参数
   - 确保架构兼容性

3. **生成参数优化**
   ```python
   outputs = model.generate(
       max_new_tokens=500,
       temperature=0.1,        # 低温度提高稳定性
       do_sample=True,
       top_p=0.9,              # Nucleus sampling
       repetition_penalty=1.15 # 抑制重复
   )
   ```
   - 平衡生成质量和速度
   - 防止重复输出

4. **内存管理优化**
   - 使用 `device_map="auto"` 自动分配
   - 显存利用率: 29.6%（14.19 GB / 47.98 GB）
   - 支持更大 batch size 和上下文长度

**性能测试结果：**

| 测试场景 | 生成 Tokens | 耗时 | 速度 |
|---------|------------|------|------|
| 短文本 | 20 | 0.75s | 26.6 tokens/s |
| 中等文本 | 50 | 2.06s | 24.3 tokens/s |
| 长文本 | 100 | 3.89s | 25.7 tokens/s |
| 代码生成 | 80 | 3.10s | 25.8 tokens/s |
| **平均** | - | - | **25.6 tokens/s** |

**Agent 任务性能：**
- 系统时间查询: 2.27s
- 知识检索问答: 6.42s
- 数学计算: 2.65s

**优化效果：**
- 推理速度: 25.6 tokens/s（7B 模型在单 GPU 上的优秀表现）
- 模型加载: 5秒（快速启动）
- 显存效率: 29.6%（留有充足空间扩展）
- 任务完成率: 100%（所有测试用例成功）

---

## 项目源代码

### 代码仓库结构

```
/workspace/EasyAgent/
├── easy_agent.py              # 主程序（520+ 行）
│   ├── MemoryManager          # 记忆管理系统
│   ├── ToolSystem             # 工具注册与执行
│   ├── EasyAgent              # ReAct Agent 核心
│   └── create_web_ui          # Gradio Web UI
├── benchmark.py               # 性能基准测试
├── models/
│   └── Qwen2.5-7B-Instruct/   # 本地模型（15GB）
├── vllm_rocm_env/             # Python 虚拟环境
├── memory.json                # 记忆持久化文件
├── benchmark_report.json      # 性能测试报告
├── README.md                  # 项目文档
└── SUBMISSION.md              # 本文档
```

### 核心代码片段

**ReAct 循环实现：**
```python
for step in range(max_steps):
    # 1. 构建 prompt（含工具定义）
    prompt = tokenizer.apply_chat_template(
        messages, tools=tool_schemas,
        tokenize=False, add_generation_prompt=True
    )
    
    # 2. 模型推理
    outputs = model.generate(**inputs, ...)
    response = tokenizer.decode(outputs[0], ...)
    
    # 3. 解析工具调用
    tool_calls = parse_function_calls(response)
    
    if tool_calls:
        # 4. 执行工具
        for tc in tool_calls:
            result = tools.execute(tc.name, tc.args)
            messages.append({"role": "tool", "content": result})
    else:
        # 5. 返回最终答案
        return clean_response(response)
```

**工具调用示例：**
```
用户: What time is it?

Agent 思考: 需要获取系统时间
Agent 行动: 