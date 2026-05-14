"""
LLM Fine-tuning Script — Conversation Drift Classification
HuggingFace Transformers + PEFT (LoRA/QLoRA)

Expects CSV files with at minimum these columns:
    window_text   — the conversation window (model input)
    drift_label   — 0 (no drift) or 1 (drift)

Install deps:
    pip install transformers peft accelerate bitsandbytes datasets scikit-learn pandas
"""

from dataclasses import dataclass, field

import mlflow
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel

)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class FinetuneConfig:
    # Model
    base_model: str = "meta-llama/Llama-3.2-1B"   # swap to any seq-cls model
    use_4bit: bool = False                           # QLoRA; set False for LoRA only

    # Data — single CSV or separate train/eval CSVs
    data_file: str = "dataset/ami/ami_full.csv"          # used when eval_file is None (auto-split)
    train_file: str = None               # set explicitly to skip auto-split
    eval_file: str = None
    eval_split: float = 0.15            # fraction held out when auto-splitting
    text_col: str = "window_text"
    topic_col: str = "agenda_item"
    label_col: str = "drift_label"
    label_names: list[str] = field(default_factory=lambda: ["no_drift", "drift"])

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Training
    output_dir: str = "./output"
    num_epochs: int = 6
    batch_size: int = 4
    grad_accum_steps: int = 4
    learning_rate: float = 3e-5
    max_length: int = 320
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 25
    fp16: bool = False                              # set False on CPU / MPS
    bf16: bool = False                             # set True if your GPU supports it (Ampere+)

    experiment_name: str = "drift-classification"



cfg = FinetuneConfig()

# ─── Data ─────────────────────────────────────────────────────────────────────

def csv_to_df(df: pd.DataFrame) -> Dataset:
    """Combine topic + conversation into a single model input."""

    df["text"] = (
        "Topic: " + df[cfg.topic_col].astype(str)
        + "\nTranscript: " + df[cfg.text_col].astype(str)
    )

    df = df[["text", cfg.label_col]].rename(
        columns={cfg.label_col: "label"}
    )

    df["label"] = df["label"].astype(int)

    return Dataset.from_pandas(df, preserve_index=False)


def make_dataset() -> DatasetDict:
    if cfg.train_file and cfg.eval_file:
        train_df = pd.read_csv(cfg.train_file)
        eval_df = pd.read_csv(cfg.eval_file)
    else:
        df = pd.read_csv(cfg.data_file)
        train_df, eval_df = train_test_split(
            df,
            test_size=cfg.eval_split,
            stratify=df[cfg.label_col],
            random_state=42,
        )
    return DatasetDict(train=csv_to_df(train_df), eval=csv_to_df(eval_df))


# ─── Tokenise ─────────────────────────────────────────────────────────────────

def tokenise(dataset: DatasetDict, tokenizer: AutoTokenizer) -> DatasetDict:
    def _tokenise(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_length,
        )

    return dataset.map(_tokenise, batched=True, remove_columns=["text"])


def check_token_lengths(dataset: DatasetDict, tokenizer: AutoTokenizer):
    lengths = [
        len(tokenizer(text)["input_ids"])
        for text in dataset["train"]["text"]
    ]
    lengths = pd.Series(lengths)
    print(lengths.describe())
    print(f"p95: {lengths.quantile(0.95):.0f}")
    print(f"p99: {lengths.quantile(0.99):.0f}")

# ─── Model ────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer():
    num_labels = len(cfg.label_names)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if cfg.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model,
        num_labels=num_labels,
        quantization_config=bnb_config,
        device_map="cuda:0",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Using device: {next(model.parameters()).device}")

    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


# ─── Train ────────────────────────────────────────────────────────────────────

def main():
    print("Loading model and tokenizer…")
    model, tokenizer = load_model_and_tokenizer()

    print("Loading and tokenising dataset…")
    raw = make_dataset()
    check_token_lengths(raw, tokenizer)
    tokenised = tokenise(raw, tokenizer)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        # gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        report_to="none",           # swap to "wandb" if you want experiment tracking
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["eval"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    mlflow.set_experiment(cfg.experiment_name)

    mlflow.log_params({
            "base_model": cfg.base_model,
            "epochs": cfg.num_epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "max_length": cfg.max_length,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
        })

    print("Starting training…")
    trainer.train()

    metrics = trainer.evaluate()
    mlflow.log_metrics(metrics)

    print(f"Saving adapter to {cfg.output_dir}/best")
    model.save_pretrained(f"{cfg.output_dir}/best")
    tokenizer.save_pretrained(f"{cfg.output_dir}/best")
    print("Done.")


# ─── Inference helper ─────────────────────────────────────────────────────────

def predict(
    topic: str,
    transcript: str,
    model_dir: str = f"{cfg.output_dir}/best",
):

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model,
        num_labels=len(cfg.label_names),
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()

    combined_text = (
        f"Topic: {topic}\n"
        f"Transcript: {transcript}"
    )

    inputs = tokenizer(
        combined_text,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.max_length,
        padding=True,
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=-1).squeeze()
    pred_idx = logits.argmax(dim=-1).item()

    return {
        "label": cfg.label_names[pred_idx],
        "drift": pred_idx == 1,
        "confidence": round(probs[pred_idx].item(), 4),
        "probabilities": {
            cfg.label_names[i]: round(probs[i].item(), 4)
            for i in range(len(cfg.label_names))
        },
    }


if __name__ == "__main__":
    main()