"""
Run inference with the fine-tuned model.
Usage: python infer.py "hey do you love cats? ... wow that is a lot lol"
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Use merged model if available, otherwise load adapter
MODEL_PATH  = "./llama-drift-merged"   # or "./llama-drift-qlora" for adapter only
MAX_SEQ_LEN = 320
LABEL_NAMES = ["no_drift", "drift"]


def predict(text: str, model, tokenizer) -> dict:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze()
    pred  = logits.argmax(-1).item()

    return {
        "label":         LABEL_NAMES[pred],
        "drift":         pred == 1,
        "confidence":    round(probs[pred].item(), 4),
        "no_drift_prob": round(probs[0].item(), 4),
        "drift_prob":    round(probs[1].item(), 4),
    }


def main():
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "hey do you love cats? i have two cats and 1000 hats for them! "
        "what is your favorite season? mine is winter!"
    )

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,          # merged model, no adapter needed
        num_labels=2,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    result = predict(text, model, tokenizer)
    print(f"\nInput   : {text[:120]}...")
    print(f"Drift   : {'YES' if result['drift'] else 'NO'}")
    print(f"Output  : {result}")


if __name__ == "__main__":
    main()
