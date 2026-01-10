import torch
import logging
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class LLMHandler:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.initialized = False
        
    def initialize(self, context):
        """Загрузка модели при старте"""
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading model from {model_dir}")
        
        # Загрузка токенизатора и модели
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        
        self.model.to(self.device)
        self.model.eval()
        self.initialized = True
        logger.info("Model loaded successfully")
        
    def preprocess(self, data):
        """Подготовка входных данных"""
        text = data[0].get("body")
        
        if isinstance(text, (bytes, bytearray)):
            text = text.decode('utf-8')
            
        # Парсим JSON
        try:
            text = json.loads(text)
        except:
            pass
                
        # Извлекаем prompt
        if isinstance(text, dict):
            prompt = text.get("prompt", text.get("text", text.get("data", "")))
            self.max_length = text.get("max_length", 200)
            self.temperature = text.get("temperature", 0.7)
            self.top_p = text.get("top_p", 0.9)
        else:
            prompt = str(text)
            self.max_length = 200
            self.temperature = 0.7
            self.top_p = 0.9
            
        logger.info(f"Processing prompt: {prompt[:100]}...")
        
        # Токенизация
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        return inputs
        
    def inference(self, inputs):
        """Генерация текста"""
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        return outputs
        
    def postprocess(self, outputs):
        """Декодирование результата"""
        generated_text = self.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
        )
        
        logger.info(f"Generated text: {generated_text[:100]}...")
        
        return [generated_text]
        
    def handle(self, data, context):
        """Главный метод обработки"""
        if not self.initialized:
            self.initialize(context)
        
        inputs = self.preprocess(data)
        outputs = self.inference(inputs)
        return self.postprocess(outputs)


# Для совместимости с TorchServe
_service = LLMHandler()

def handle(data, context):
    return _service.handle(data, context)