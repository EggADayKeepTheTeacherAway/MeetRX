"""
Preprocessor for AMI / ICSI topic JSON files.
Produces agenda-grounded drift detection training data.

Output schema:
    meeting_id      e.g. ES2002
    session_id      e.g. ES2002a
    agenda_item     topic label (e.g. "budget")
    window_text     "SPEAKER: utterance ..." for the 50s window
    time_start      window start in seconds
    time_end        window end in seconds
    drift_label     0 = on agenda, 1 = drifted

Usage:
    python preprocess_ami.py --ami_dir topic/ami --icsi_dir topic/icsi \
                             --output_dir dataset --window_sec 50 \
                             --stride_sec 25 --eval_split 0.15 --seed 42
"""

import argparse
import glob
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ─── Agenda-item labels excluded from positive (drift=0) classes ─────────────
# Windows from these topics are never used as positives for any agenda class.
# They are pooled as negatives and paired with a random agenda_item → drift=1.
EXCLUDED_AGENDA_ITEMS = {"discussion", "closing", "chitchat", "opening"}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def meeting_id(session_id: str) -> str:
    """ES2002a → ES2002"""
    return re.sub(r"[a-z]$", "", session_id)


def clean_text(text: str) -> str:
    """Strip ASR noise tokens like <vocalsound>, <disfmarker> etc."""
    text = re.sub(r"<[^>]+>", "", text)
    return " ".join(text.split()).strip()


# ─── Parse ────────────────────────────────────────────────────────────────────

def parse_session(path: Path) -> dict:
    """
    Returns:
        {
          session_id: str,
          topics: [
            {
              topic: str,
              is_other: bool,
              acts: [{speaker, starttime, endtime, text}]
            }
          ]
        }
    """
    data = json.loads(path.read_text())
    session_id = path.stem          # ES2002a

    topics = []
    raw = data if isinstance(data, list) else [data]

    for topic_block in raw:
        label = topic_block.get("topic", "other").strip().lower()
        acts = []
        for da in topic_block.get("dialogueacts", []):
            text = clean_text(da.get("text", ""))
            if not text:
                continue
            acts.append({
                "speaker":   da.get("speaker", "?"),
                "starttime": float(da.get("starttime", 0)),
                "endtime":   float(da.get("endtime", 0)),
                "text":      text,
            })
        if acts:
            topics.append({
                "topic":      label,
                "is_other":   label in {"other", "none"},
                "is_excluded": label in EXCLUDED_AGENDA_ITEMS,
                "acts":       sorted(acts, key=lambda x: x["starttime"]),
            })

    return {"session_id": session_id, "topics": topics}


# ─── Window builder ───────────────────────────────────────────────────────────

def build_windows(acts: list[dict], window_sec: float, stride_sec: float) -> list[dict]:
    """
    Slide a fixed-duration window over a list of dialogue acts using stride_sec
    so consecutive windows overlap. A window is kept only if it contains at
    least 30 tokens (words) to filter out sparse segments.

    Returns list of {time_start, time_end, window_text}.
    """
    if not acts:
        return []

    MIN_CHARS  = 100
    MIN_TOKENS = 30

    windows = []
    start = acts[0]["starttime"]
    end_of_acts = acts[-1]["endtime"]

    while start < end_of_acts:
        end = start + window_sec
        chunk = [a for a in acts if a["starttime"] >= start and a["endtime"] <= end]
        if chunk:
            text = " ".join(f"{a['speaker']}: {a['text']}" for a in chunk)
            # ── Step 1 fix: drop windows that are too sparse ──────────────────
            if len(text) >= MIN_CHARS and len(text.split()) >= MIN_TOKENS:
                windows.append({
                    "time_start":  round(start, 2),
                    "time_end":    round(end, 2),
                    "window_text": text,
                })
        # ── Step 1 fix: advance by stride, not by full window ─────────────────
        start += stride_sec

    return windows


# ─── Main build ───────────────────────────────────────────────────────────────

def build_dataset(
    sessions: list[dict],
    window_sec: float,
    stride_sec: float,
    seed: int,
) -> pd.DataFrame:
    """
    For every non-excluded topic T in every session:
      - Positive rows  : windows from T, agenda_item=T.topic          → label 0
      - Negative rows  : windows from wrong-topic segments
                         + windows from "other" segments
                         + windows from EXCLUDED_AGENDA_ITEMS,
                           each paired with a random real agenda_item  → label 1
    """
    rng = random.Random(seed)

    # ── collect windows per topic type ────────────────────────────────────────
    topic_windows: dict[str, list[dict]] = defaultdict(list)   # real agenda topics
    other_windows: list[dict] = []                              # other/none
    excluded_windows: list[dict] = []                           # discussion/closing/etc.

    for sess in sessions:
        sid = sess["session_id"]
        mid = meeting_id(sid)
        for t in sess["topics"]:
            wins = build_windows(t["acts"], window_sec, stride_sec)
            for w in wins:
                w["session_id"] = sid
                w["meeting_id"] = mid
                w["topic"]      = t["topic"]

            if t["is_excluded"]:
                excluded_windows.extend(wins)
            elif t["is_other"]:
                other_windows.extend(wins)
            else:
                topic_windows[t["topic"]].extend(wins)

    real_agenda_labels = list(topic_windows.keys())

    print(f"  excluded agenda windows : {len(excluded_windows)}"
          f"  ({', '.join(sorted(EXCLUDED_AGENDA_ITEMS))})")
    print(f"  other/none windows      : {len(other_windows)}")

    rows = []

    for topic_label, pos_windows in topic_windows.items():
        if not pos_windows:
            continue

        # ── positives ─────────────────────────────────────────────────────────
        for w in pos_windows:
            rows.append({
                "meeting_id":  w["meeting_id"],
                "session_id":  w["session_id"],
                "agenda_item": topic_label,
                "window_text": w["window_text"],
                "time_start":  w["time_start"],
                "time_end":    w["time_end"],
                "drift_label": 0,
            })

        n_pos = len(pos_windows)

        # ── negatives: wrong topic windows ────────────────────────────────────
        wrong_topic_pool = [
            w for lbl, wins in topic_windows.items()
            if lbl != topic_label
            for w in wins
        ]

        # ── negatives: combined pool (wrong-topic + other) ────────────────────
        neg_pool = wrong_topic_pool + other_windows

        neg_sample = rng.sample(neg_pool, min(n_pos, len(neg_pool)))

        for w in neg_sample:
            rows.append({
                "meeting_id":  w["meeting_id"],
                "session_id":  w["session_id"],
                "agenda_item": topic_label,
                "window_text": w["window_text"],
                "time_start":  w["time_start"],
                "time_end":    w["time_end"],
                "drift_label": 1,
            })

    # ── negatives from excluded agenda items ──────────────────────────────────
    # Cap to total positives so excluded windows don't inflate drift ratio.
    n_pos_total = sum(1 for r in rows if r["drift_label"] == 0)
    n_neg_so_far = sum(1 for r in rows if r["drift_label"] == 1)
    n_excluded_budget = max(0, n_pos_total - n_neg_so_far)

    rng.shuffle(excluded_windows)
    for w in excluded_windows[:n_excluded_budget]:
        assigned_agenda = rng.choice(real_agenda_labels)
        rows.append({
            "meeting_id":  w["meeting_id"],
            "session_id":  w["session_id"],
            "agenda_item": assigned_agenda,
            "window_text": w["window_text"],
            "time_start":  w["time_start"],
            "time_end":    w["time_end"],
            "drift_label": 1,
        })

    df = pd.DataFrame(rows)

    # ── enforce global 50/50 by downsampling the majority class ───────────────
    pos = df[df["drift_label"] == 0]
    neg = df[df["drift_label"] == 1]
    n   = min(len(pos), len(neg))
    df  = pd.concat([
        pos.sample(n, random_state=seed),
        neg.sample(n, random_state=seed),
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"  balanced total : {len(df)}  drift ratio: {df['drift_label'].mean():.2f}")
    return df


# ─── Split ────────────────────────────────────────────────────────────────────

def split_dataset(
    df: pd.DataFrame,
    eval_split: float,
    seed: int,
    output_dir: Path,
    prefix: str,
):
    """Group-split by meeting_id so no meeting leaks across train/eval."""
    gss = GroupShuffleSplit(n_splits=1, test_size=eval_split, random_state=seed)
    train_idx, eval_idx = next(gss.split(df, groups=df["meeting_id"]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    eval_df  = df.iloc[eval_idx].reset_index(drop=True)

    train_path = output_dir / f"{prefix}_train.csv"
    eval_path  = output_dir / f"{prefix}_eval.csv"

    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path,  index=False)

    print(f"[{prefix}] train={len(train_df)}  eval={len(eval_df)}")
    print(f"  drift ratio train: {train_df['drift_label'].mean():.2f}")
    print(f"  drift ratio eval : {eval_df['drift_label'].mean():.2f}")
    print(f"  → {train_path}")
    print(f"  → {eval_path}")

    df.to_csv(output_dir / f"{prefix}_full.csv", index=False)
    print(f"[{prefix}] total={len(df)}  drift ratio: {df['drift_label'].mean():.2f}")
    print(f"  → {output_dir / f'{prefix}_full.csv'}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ami_dir",    type=str,   default="topics")
    parser.add_argument("--icsi_dir",   type=str,   default=None)
    parser.add_argument("--output_dir", type=str,   default="dataset")
    parser.add_argument("--window_sec", type=float, default=50.0)
    parser.add_argument("--stride_sec", type=float, default=25.0)   # NEW arg
    parser.add_argument("--eval_split", type=float, default=0.15)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {}
    if args.ami_dir:
        datasets["ami"] = args.ami_dir
    if args.icsi_dir:
        datasets["icsi"] = args.icsi_dir

    if not datasets:
        raise ValueError("Provide at least one of --ami_dir or --icsi_dir")

    for name, dir_path in datasets.items():
        files = glob.glob(f"{dir_path}/*.json")
        if not files:
            print(f"[{name}] No JSON files found in {dir_path}, skipping.")
            continue

        print(f"\n[{name}] Found {len(files)} session files...")
        sessions = [parse_session(Path(f)) for f in files]

        df = build_dataset(sessions, args.window_sec, args.stride_sec, args.seed)
        print(f"[{name}] Total rows: {len(df)}")
        print(f"  Topics found: {sorted(df['agenda_item'].unique())}")

        split_dataset(df, args.eval_split, args.seed, output_dir, name)

    print("\nDone.")


if __name__ == "__main__":
    main()