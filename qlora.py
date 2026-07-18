from Transformers import AutoModel
from peft import LoraModel, LoraConfig

model = AutoModel.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", dtype="auto") 
