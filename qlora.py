from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
import torch

#LORA CONFIGURATION + TRANSFORMER + TO TRAIN MODEL

peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
#QUANTIZATION
quantization_config = BitsAndBytesConfig(load_in_4bit=True,
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16, 
    bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B", dtype="auto", quantization_config=quantization_config) 
model = get_peft_model(model, peft_config)

configurations = peft_config.to_dict()
print(f"Lora configurations: {configurations}, model trainable parametersL: {model.print_trainable_parameters()}")

#LOAD DATASET
dataset = load_dataset("json", data_files="./pathtodegradedcorpus")

training_args = SFTConfig(packing=True, dataset_text_field="content") #PACKING IS IDEAL FOR CONTINUOUS PRE TRAINING

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)