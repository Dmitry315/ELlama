#!/bin/bash

MODEL_PATH="data/model" 
PORT=8000
API_KEY="kek"

echo "Запуск vLLM сервера на одном GPU..."
echo "Модель: $MODEL_PATH"
echo "Порт: $PORT"

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

VLLM_USE_V1=1 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --host 0.0.0.0 \
    --port $PORT \
    --api-key $API_KEY \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --max-model-len 1024 \
    --dtype bfloat16 \
    --served-model-name "ellama" \
    --disable-frontend-multiprocessing \
    --enforce-eager