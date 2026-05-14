# Dataset Preparation

The AMI corpus preprocessing pipeline relies on the tooling provided by
[guokan-shang/ami-and-icsi-corpora](https://github.com/guokan-shang/ami-and-icsi-corpora).
Credit to [@guokan-shang](https://github.com/guokan-shang) for the corpus extraction scripts.

---

## Prerequisites

Clone the upstream repository and install its dependencies:

```bash
git clone https://github.com/guokan-shang/ami-and-icsi-corpora.git
cd ami-and-icsi-corpora
# See the repository's README for how to install dependencies
```

---

## Steps

### 1. Extract Dialogue Acts and Topics

From inside the cloned repository, run the extraction scripts in order:

```bash
python dialogueActs.py
python topics.py
```

Output files will be written to the `output/` directory.

### 2. Run the Preprocessor

Copy `preprocess_ami.py` from this directory into the `output/` of `ami-and-icsi-corpora` directory, then run it:

```bash
python preprocess_ami.py
```

This will produce a `dataset/` directory containing the train, eval, and full CSV splits.

### 3. Copy the Dataset

Copy `ami_full.csv` into this project's dataset directory:

---

## Output Schema

| Column | Description |
|---|---|
| `meeting_id` | Meeting identifier (e.g. `ES2002`) |
| `session_id` | Session identifier (e.g. `ES2002a`) |
| `agenda_item` | Topic label |
| `window_text` | `SPEAKER: utterance ...` formatted transcript window |
| `time_start` | Window start in seconds |
| `time_end` | Window end in seconds |
| `drift_label` | `0` = on topic, `1` = drifted |