# MeetRX — Topic Drift Detection POC

A proof-of-concept system for detecting **topic drift** in meeting transcripts using a fine-tuned LLaMA 3.2 model with LoRA adapters, trained on the AMI Meeting Corpus.

---

## Project Structure

```
POC/
├── topic_drift/
│   └── training/
│       ├── dataset/
│       │   ├── ami/                      # AMI meeting corpus dataset
│       │   ├── aggregated/               # Combined dataset splits
│       │   ├── dialseg/                  # DialSeg711 dataset
│       │   ├── tiage/                    # TIAGE dataset
│       │   └── dataset_aggregate.py      # Dataset merging script
│       ├── llama-drift-merged/           # Merged LoRA model weights
│       ├── mlruns/                       # MLflow experiment tracking
│       ├── output/                       # Training checkpoints
│       ├── finetune.py                   # LoRA fine-tuning script
│       ├── infer_server.py               # API inference endpoint
│       ├── infer.py                      # CLI inference script
│       ├── merge.py                      # Merge LoRA adapter into base model
│       ├── mlflow.db                     # MLflow local database
│       └── requirements.txt             # Training dependencies
├── main.py                               # Streamlit demo application
├── requirements.txt                      # App dependencies
├── README.md
└── .gitignore
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Pipeline

## Setup

```bash
pip install -r topic_drift/training/requirements.txt
```

### 1. Fine-tune

```bash
python topic_drift/training/finetune.py
```

Trains a LoRA adapter on top of `meta-llama/Llama-3.2-1B` for binary sequence classification (`no_drift` / `drift`). Experiment metrics are logged to MLflow.

### 2. Merge Adapter

```bash
python topic_drift/training/merge.py
```

Merges the LoRA adapter into the base model and saves full weights to `llama-drift-merged/`.

### 3. CLI Inference

```bash
python topic_drift/training/infer.py "budget discussion" "A: Let's go over Q3 spending. B: We're over by 10k."
```

### 4. Streamlit Demo

```bash
streamlit run main.py
```

---

## Model

| Property | Value |
|---|---|
| Base model | `meta-llama/Llama-3.2-1B` |
| Method | LoRA (PEFT) |
| Task | Binary sequence classification |
| Labels | `no_drift` (0), `drift` (1) |
| Max sequence length | 320 tokens |
| LoRA rank | 32 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |

---

## Dataset

Training data is sourced from three corpora and merged into a unified schema:

| Source | Description |
|---|---|
| AMI Meeting Corpus | Product design meetings, annotated topic segments |
| DialSeg711 | Dialogue segmentation benchmark  |
| TIAGE | Topic-annotated dialogue dataset |

Each row in the dataset represents a 50-second sliding window of meeting transcript with the following schema:

| Column | Description |
|---|---|
| `meeting_id` | Meeting identifier |
| `session_id` | Session identifier |
| `agenda_item` | Topic label |
| `window_text` | `SPEAKER: utterance ...` formatted transcript window |
| `time_start` | Window start in seconds |
| `time_end` | Window end in seconds |
| `drift_label` | `0` = on topic, `1` = drifted |

---

## Demo App

The Streamlit app (`main.py`) provides:

- **Per-turn transcript editor** — each speaker turn is an individual editable row
- **🎲 Random sample** — pulls a window from `ami_full.csv` with its true label
- **Test Scenario 1–3** — pre-loaded transcripts covering on-topic and drift cases
- **Inference result card** — shows verdict, confidence, and per-class probabilities

---

## Demo API endpoint

Run the `infer_server.py` script directly.

```bash
python topic_drift/training/infer_server.py
```

---

## Experiment Tracking

MLflow is used for tracking training runs locally

---

## Requirements

- Python 3.11
- PyTorch with CUDA (recommended) or CPU
