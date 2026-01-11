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

# DVC

Использую dagshub:

https://dagshub.com/melikhov.dmitry.a/ELlama

# Serve

В CI/CD некоторые библиотеки не влезли по памяти, поэтому в следующих частях пишется что нужно доустановить.

По умолчанию у претрен модели нет chat_template, поэтому для правильной работы добавляем просто конкатенацию.
Скачиваем модель локально и добавляем шаблон:
```
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "dmitry315/ELlama1-0.7b"
save_path = "data/model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

chat_template = """{% for message in messages %}{{ message['content'] }}{% endfor %}"""

tokenizer.chat_template = chat_template

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
```

## VLLM (просто + просто)

Install:
```
pip install vllm==0.13.0 openai
```

Запускаем serve (`run_vllm_serve.sh`)

Запросы к модели через OpenAI API:

```
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="kek"
)

response = client.chat.completions.create(
    model="ellama",
    messages=[
        {"role": "user", "content": "Θυμάμαι μια υπέροχη"}
    ],
    temperature=0.7,
    max_tokens=100
)

print(response.choices[0].message.content)
```

Ещё есть возможность обернуть vllm serve в красивый web-ui:
- https://github.com/open-webui/open-webui
- https://docs.openwebui.com/getting-started/quick-start/starting-with-vllm

## TorchServe (тяжело + зачем?)

Install:
```
pip install torchserve torch-model-archiver torch-workflow-archiver 
```

Архивируем (`run_torch_archiver.sh`)

Запускаем (`run_torchserve.sh`)

Делаем запрос:
```
curl -X POST http://localhost:8080/predictions/ellama \
  -H "Content-Type: application/json" \
  -d '{"data": "Θυμάμαι μια υπέροχη"}'
```

# MLFlow

Логи эксперимента лежат в DagsHub: https://dagshub.com/melikhov.dmitry.a/ELlama/experiments

Например: https://dagshub.com/melikhov.dmitry.a/ELlama/experiments#/experiment/m_5f05459c5b854cee9c30cea1e1dbfdbc
