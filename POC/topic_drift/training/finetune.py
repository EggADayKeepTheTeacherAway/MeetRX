"""
Fine-tune Llama 3.1 8B (QLoRA) on topic drift detection
Usage: python finetune.py
"""

import os
import torch
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ─────────────────────────────────────────────
# CONFIG — tune these based on your GPU memory
# ─────────────────────────────────────────────
MODEL_ID       = "meta-llama/Llama-3.1-8B"
DATA_PATH      = "windows.csv"
OUTPUT_DIR     = "./llama-drift-qlora"
HF_TOKEN       = os.environ.get("HF_TOKEN", "")   # set via: export HF_TOKEN=hf_...

MAX_SEQ_LEN    = 512    # lower to 256 if OOM
BATCH_SIZE     = 2      # lower to 1 if OOM
GRAD_ACCUM     = 8      # effective batch = BATCH_SIZE * GRAD_ACCUM
EPOCHS         = 3
LR             = 2e-4
LORA_R         = 16     # lower to 8 if OOM
LORA_ALPHA     = 32
VAL_SPLIT      = 0.15
SEED           = 42
# ─────────────────────────────────────────────


def format_prompt(row) -> str:
    """Convert a windows.csv row into an instruction prompt."""
    label = "yes" if row["drift_label"] == 1 else "no"
    return (
        "### Conversation:\n"
        f"{row['window_text']}\n\n"
        "### Does this conversation contain a topic shift?\n"
        f"{label}"
    )


def load_dataset(path: str):
    df = pd.read_csv(path)
    df = df.dropna(subset=["window_text", "drift_label"])
    df["drift_label"] = df["drift_label"].astype(int)

    print(f"\nDataset: {len(df)} windows")
    print(f"  Drifted : {df['drift_label'].sum()} ({df['drift_label'].mean()*100:.1f}%)")
    print(f"  No drift: {(~df['drift_label'].astype(bool)).sum()}\n")

    df["text"] = df.apply(format_prompt, axis=1)

    train_df, val_df = train_test_split(
        df, test_size=VAL_SPLIT, random_state=SEED, stratify=df["drift_label"]
    )

    train_ds = Dataset.from_pandas(train_df[["text"]].reset_index(drop=True))
    val_ds   = Dataset.from_pandas(val_df[["text", "drift_label"]].reset_index(drop=True))

    return train_ds, val_ds, val_df


def load_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, token=HF_TOKEN, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def evaluate(model, tokenizer, val_df: pd.DataFrame):
    """Run greedy decode on val set and compute F1."""
    model.eval()
    preds, labels = [], []

    for _, row in val_df.iterrows():
        prompt = (
            "### Conversation:\n"
            f"{row['window_text']}\n\n"
            "### Does this conversation contain a topic shift?\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=MAX_SEQ_LEN).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=3, do_sample=False)
        decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip().lower()
        pred = 1 if "yes" in decoded else 0
        preds.append(pred)
        labels.append(int(row["drift_label"]))

    print("\n── Evaluation ──────────────────────────")
    print(classification_report(labels, preds, target_names=["no drift", "drift"]))
    print(f"Macro F1: {f1_score(labels, preds, average='macro'):.4f}")
    print("────────────────────────────────────────\n")


def main():
    print("Loading data...")
    train_ds, val_ds, val_df = load_dataset(DATA_PATH)

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()

    # Only compute loss on the answer token(s), not the prompt
    response_template = "### Does this conversation contain a topic shift?\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template, tokenizer=tokenizer
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",           # change to "wandb" if you want tracking
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
    )

    print("Starting training...")
    trainer.train()

    print("Saving LoRA adapter...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Running evaluation...")
    evaluate(model, tokenizer, val_df)

    print(f"\nDone. Adapter saved to: {OUTPUT_DIR}")
    print("To merge weights later, run: python merge.py")


if __name__ == "__main__":
    main()
