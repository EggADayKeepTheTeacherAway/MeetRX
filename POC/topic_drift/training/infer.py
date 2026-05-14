"""
Run inference with the fine-tuned model.
Usage: python infer.py "<topic>" "<transcript>"

Example:
    python infer.py "budget discussion" "A: Let's go over Q3 spending. B: We're over by 10k."
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH  = "./llama-drift-merged"
MAX_SEQ_LEN = 320
LABEL_NAMES = ["no_drift", "drift"]


def predict(topic: str, text: str, model, tokenizer) -> dict:
    combined_text = (
        f"Topic: {topic}\n"
        f"Transcript: {text}"
    )

    inputs = tokenizer(
        combined_text,
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
    if len(sys.argv) >= 3:
        topic = sys.argv[1]
        text  = " ".join(sys.argv[2:])
    elif len(sys.argv) == 2:
        print("Usage: python infer.py \"<topic>\" \"<transcript>\"")
        sys.exit(1)
    else:
        # default demo
        topic = "budget discussion"
        text  = (
            "A: Let's go over Q3 spending. "
            "B: We're over by ten thousand. "
            "A: We need to cut the hardware order. "
            "C: Did anyone watch the game last night?"
        )

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=2,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    result = predict(topic, text, model, tokenizer)
    print(f"\nTopic   : {topic}")
    print(f"Input   : {text[:120]}...")
    print(f"Drift   : {'YES' if result['drift'] else 'NO'}")
    print(f"Output  : {result}")


if __name__ == "__main__":
    main()