"""
Run inference with the fine-tuned model.
Usage: python infer.py "hey do you love cats? ... wow that is a lot lol"
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Use merged model if available, otherwise load adapter
MODEL_PATH  = "./llama-drift-merged"   # or "./llama-drift-qlora" for adapter only
MAX_SEQ_LEN = 512


def predict(text: str, model, tokenizer) -> dict:
    prompt = (
        "### Conversation:\n"
        f"{text}\n\n"
        "### Does this conversation contain a topic shift?\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=MAX_SEQ_LEN).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    answer = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip().lower()
    return {
        "drift": "yes" in answer,
        "raw_output": answer,
    }


def main():
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "hey do you love cats? i have two cats and 1000 hats for them! "
        "what is your favorite season? mine is winter!"
    )

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    result = predict(text, model, tokenizer)
    print(f"\nInput   : {text[:120]}...")
    print(f"Drift   : {'YES' if result['drift'] else 'NO'}")
    print(f"Output  : {result['raw_output']}")


if __name__ == "__main__":
    main()
