"""
Merge LoRA adapter into base model and save full weights.
Usage: python merge.py
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

BASE_MODEL  = "meta-llama/Llama-3.2-1B"
ADAPTER_DIR = "./output/best/"
OUTPUT_DIR  = "./llama-drift-merged"
HF_TOKEN    = os.environ.get("HF_TOKEN", "")

print("Loading base model...")
base = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=2,               # must match cfg.label_names length
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    token=HF_TOKEN,
)


print("Loading adapter...")
model = PeftModel.from_pretrained(base, ADAPTER_DIR)

print("Merging weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR, max_shard_size="2GB")

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Done.")
