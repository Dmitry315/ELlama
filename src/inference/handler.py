# handler.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler

class LLMHandler(BaseHandler):
    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        # Загрузка модели и токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        
    def preprocess(self, data):
        text = data[0].get("content") or data[0]
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        inputs = self.tokenizer(text, return_tensors="pt")
        return inputs
        
    def inference(self, inputs):
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=1024,
                temperature=0.1,
                top_p=10
            )
        return outputs
        
    def postprocess(self, outputs):
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [text]