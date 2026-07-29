#!/bin/bash
# start_server.sh - 启动包含深度硬件优化的 vLLM API 服务器

echo "应用 AMD ROCm 性能调优层..."

export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
export HIP_FORCE_DEV_KERNARG=1
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=1
# 更新 CSV 文件的生成路径到当前工程下
export PYTORCH_TUNABLEOP_FILENAME="/workspace/EasyAgent/tunableop_qwen25_7b.csv"
export HSA_OVERRIDE_GFX_VERSION="11.0.0"

# 更新为实际的模型挂载绝对路径
MODEL_PATH="/workspace/EasyAgent/models/Qwen2.5-7B-Instruct" 
echo "正在针对模型 $MODEL_PATH 启动后端引擎..."

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "Qwen/Qwen2.5-7B-Instruct" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --port 8000