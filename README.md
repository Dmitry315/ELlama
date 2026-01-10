# ElladaGPT

Цель проекта - создать чат бот с нуля... на греческом языке.

# К чему стремимся

Точность на открытом бенчмарке (перевод): > 75%

Скорость отклика модели: < 1000мс

# Набор данных

## Pretrain

Для предобучения будут использоваться данные fineweb2 на греческом: 
- Современный (~70GB parquet, ~200MB utf-8 байтов, ~22B слов)[fineweb-2/viewer/ell_Grek](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/viewer/ell_Grek)

## SFT

Для SFT стадии будет использоваться:
- GPT-4 синта на греческом: [CausalLM/GPT-4-Self-Instruct-Greek](https://huggingface.co/datasets/CausalLM/GPT-4-Self-Instruct-Greek)
- instruct датсет (перевод почищенной альпаки от Стенфорда): [iamshnoo/alpaca-cleaned-greek](https://huggingface.co/datasets/iamshnoo/alpaca-cleaned-greek)

## Bench

Перевод mt-бенча на греческий: [ilsp/mt-bench-greek](https://huggingface.co/datasets/ilsp/mt-bench-greek)

# План технической реализации

Для этого проекта потребуется обучить токенизатор и модель, а также научиться бытро их инферить и считать качество на незнакомом языке.

Обучение BPE токенизатора происходит через библиотеку tokenizers. Для эффективного инференса лучше всего использовать tiktoken, так как он гораздо быстрее чем HF токенизаторы.

Планируется обучить 1-7b модель с использованием библиотек TRL (для SFT) и Accelerate (для FSDP).

Нужно разработать бенчмарк, где оценка считается через перевод запроса и оценки перевода ответа (LLM as a judge).

Быстрый инференс может осуществлятся через библиотеку vllm.

# Quick start

Делаем окружение с питоном 3.10 ставим requirements.txt
```
conda create --name ellama-learn python=3.10 --yes
conda activate ellama-learn
pip install -r requirements.txt
pip install -e .
```

Скачиваем данные для обучения:
```
sh scripts/download_fineweb.sh
```

Запускаем обучение токенизатора:
```
sh scripts/run_tokenizer_train.sh
```

Запускаем обучение модели:
```
sh scripts/run_model_train.sh
```

# Serve

## TorchServe

Скачиваем модель локально:
```
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "dmitry315/ELlama1-0.7b"
save_path = "data/model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
```

Архивируем (`run_torch_archiver.sh`):
```
torch-model-archiver \
  --model-name llm_text_generation \
  --version 1.0 \
  --handler llm_handler.py \
  --extra-files "./data/model" \
  --export-path model_store \
  --force
```

Запускаем (`run_torchserve.sh`):
```
torchserve --start \
  --model-store model_store \
  --models ellama=llm_text_generation.mar \
  --ncs
```

Делаем запрос:
```
curl -X POST http://localhost:8080/predictions/ellama \
  -H "Content-Type: application/json" \
  -d '{"data": "Θυμάμαι μια υπέροχη"}'
```

# MLFlow

Логи эксперимента лежат в DagsHub: https://dagshub.com/melikhov.dmitry.a/ellama-train/experiments

Например: https://dagshub.com/melikhov.dmitry.a/ellama-train/experiments#/experiment/m_7978b647680e4765ac9ba4e0bd0be50f
