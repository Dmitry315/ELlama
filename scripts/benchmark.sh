OPENAI_API_KEY="kek" \
vllm bench serve \
  --backend openai-chat \
  --base-url http://localhost:8000 \
  --endpoint /v1/chat/completions \
  --model gpt2 \
  --served-model-name ellama \
  --dataset-name random \
  --num-prompts 50 \
  --random-input-len 128 \
  --random-output-len 128 \
  --request-rate 2 \
  --percentile-metrics ttft,tpot,e2el \
  --metric-percentiles 50,90,95,99 \
  --ready-check-timeout-sec 30