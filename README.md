# EasyAgent — AMD Radeon GPU Intelligent Agent System

🏆 **Hackathon Track 2: Agentic AI**

## Overview

EasyAgent is an intelligent AI agent system running locally on AMD Radeon GPUs. It demonstrates a full **ReAct (Reasoning + Acting)** loop with multi-tool calling, dual-layer memory management, and AMD ROCm-optimized inference.

### Key Features

- **ReAct Reasoning Loop** — Think → Act → Observe → Repeat
- **Native Function Calling** — Powered by Qwen2.5 chat-template-based tool invocation
- **8 Built-in Tools** — Code execution, file I/O, knowledge retrieval, GPU monitoring, and more
- **Dual-Layer Memory** — Short-term memory with automatic consolidation into long-term memory
- **AMD ROCm Optimization** — Local inference on RDNA3 GPUs (~25 tokens/s)
- **Gradio Web UI** — Friendly interactive interface

## Hardware Requirements

| Component | Specification |
|-----------|---------------|
| GPU       | AMD Radeon (gfx1100, RDNA 3) |
| Compute Units | 96 CU |
| VRAM      | 48 GB |
| CPU       | AMD EPYC 9334 32-Core |
| ROCm      | 6.2 |
| PyTorch   | 2.5.1+rocm6.2 |

## Quick Start

### 1. Environment Setup

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set AMD GPU environment variables
export HSA_OVERRIDE_GFX_VERSION="11.0.0"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

### 2. Download Model

The model weights are not included in this repository due to size (~15 GB).
Download [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) from Hugging Face and place it under `models/`:

```bash
mkdir -p models
# Option A: using huggingface-cli
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct

# Option B: using git-lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct models/Qwen2.5-7B-Instruct
```

### 3. Run the Agent

**Interactive mode:**
```bash
python3 easy_agent.py
```

**Demo mode:**
```bash
python3 easy_agent.py --demo
```

**Web UI mode:**
```bash
python3 easy_agent.py --web
# Open http://localhost:7860
```

**REST API mode:**
```bash
python3 api_server.py
# API docs at http://localhost:8000/docs
```

## Available Tools

| Tool | Description |
|------|-------------|
| `execute_code` | Execute Python code, return stdout |
| `system_info` | Get system time, GPU, and memory info |
| `read_file` | Read file contents |
| `write_file` | Write content to a file |
| `list_dir` | List directory contents |
| `search_knowledge` | Retrieve from local knowledge base |
| `gpu_monitor` | GPU utilization and temperature monitoring |
| `calculate` | Safe mathematical expression evaluation |

## Performance Benchmarks

| Metric | Result |
|--------|--------|
| **Average Inference Speed** | **25.6 tokens/s** |
| Short text (20 tokens) | 26.6 tokens/s |
| Medium text (50 tokens) | 24.3 tokens/s |
| Long text (100 tokens) | 25.7 tokens/s |
| Code generation (80 tokens) | 25.8 tokens/s |
| **Model Load Time** | **5.0 s** |
| **VRAM Usage** | **14.19 GB / 47.98 GB (29.6%)** |

### Agent Task Latency
- System time query: 2.27 s
- Knowledge retrieval QA: 6.42 s
- Math computation: 2.65 s

## Architecture

```
┌─────────────────────────────────────┐
│         EasyAgent Core              │
│  ┌──────────────────────────────┐   │
│  │      ReAct Loop              │   │
│  │  Think → Act → Observe       │   │
│  └──────────────────────────────┘   │
│                                     │
│  ┌──────────┐    ┌──────────┐      │
│  │  Memory  │    │  Tools   │      │
│  │ Short+LT │    │ 8 Tools  │      │
│  └──────────┘    └──────────┘      │
│                                     │
│  ┌──────────────────────────────┐   │
│  │  Qwen2.5-7B (Function Call)  │   │
│  │  AMD Radeon GPU + ROCm 6.2   │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
```

## AMD GPU Optimization

### ROCm Tuning

```bash
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
export HIP_FORCE_DEV_KERNARG=1
export PYTORCH_TUNABLEOP_ENABLED=1
export HSA_OVERRIDE_GFX_VERSION="11.0.0"
```

### Optimization Highlights

1. **FP16 Inference** — Half-precision for faster throughput
2. **TunableOp** — Automatic PyTorch operator tuning
3. **Memory Management** — Optimized GPU memory allocation
4. **RDNA3 Architecture** — Leveraging latest AMD GPU features

## Project Structure

```
EasyAgent/
├── easy_agent.py          # Main agent (ReAct loop, tools, Web UI)
├── api_server.py          # FastAPI REST API server
├── benchmark.py           # Performance benchmark suite
├── run_agent.py           # Alternative OpenAI-compatible runner
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT License
├── README.md              # This file
├── SUBMISSION.md          # Hackathon submission notes
├── scripts/
│   └── start_server.sh    # vLLM server launch script
└── models/                # Model weights (download separately)
    └── Qwen2.5-7B-Instruct/
```

## Tech Stack

- **Model:** Qwen2.5-7B-Instruct
- **Framework:** PyTorch 2.5.1 + ROCm 6.2
- **Inference:** Transformers + Native Function Calling
- **Web UI:** Gradio
- **API:** FastAPI + Uvicorn
- **GPU:** AMD Radeon gfx1100 (RDNA 3)

## References

- [AMD ROCm](https://www.amd.com/en/graphics/servers-solutions-rocm)
- [Qwen2.5 Models](https://qwenlm.github.io/)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [Function Calling with Qwen2.5](https://huggingface.co/blog/norod78-qwen2.5-function-calling)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ for AMD Hackathon 2026**
