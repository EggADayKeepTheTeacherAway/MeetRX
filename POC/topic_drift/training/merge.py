"""
Merge LoRA adapter into base model and save full weights.
Usage: python merge.py
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL  = "meta-llama/Llama-3.1-8B"
ADAPTER_DIR = "./llama-drift-qlora"
OUTPUT_DIR  = "./llama-drift-merged"
HF_TOKEN    = os.environ.get("HF_TOKEN", "")

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=HF_TOKEN,
)

print("Loading adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)

print("Merging weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Done.")
