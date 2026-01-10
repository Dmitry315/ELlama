torch-model-archiver \
  --model-name llm_text_generation \
  --version 1.0 \
  --handler src/inference/handler.py \
  --extra-files "data/model" \
  --export-path model_store \
  --force