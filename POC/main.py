"""
Streamlit POC – Topic Drift Detector
Usage:
    streamlit run app.py

Expects:
    - ./llama-drift-merged/   merged model weights
    - ./dataset/ami_full.csv  or icsi_full.csv  (for random sample button)

Install:
    pip install streamlit transformers peft torch pandas
"""

import random
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "./topic_drift/training/llama-drift-merged"
MAX_SEQ_LEN = 320
LABEL_NAMES = ["no_drift", "drift"]

# Looks for any *_full.csv in ./dataset/
DATASET_GLOB = "./topic_drift/training/dataset/ami/ami_full.csv"

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Topic Drift Detector",
    page_icon="📡",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif;
}

.main {
    background: #0d0d0f;
    color: #e8e6e0;
}

section[data-testid="stSidebar"] {
    background: #111114;
}

/* Result cards */
.drift-yes {
    background: linear-gradient(135deg, #1a0a0a 0%, #2d0f0f 100%);
    border: 1px solid #c0392b;
    border-left: 4px solid #e74c3c;
    border-radius: 8px;
    padding: 20px 24px;
    margin-top: 16px;
}

.drift-no {
    background: linear-gradient(135deg, #0a1a10 0%, #0f2d18 100%);
    border: 1px solid #27ae60;
    border-left: 4px solid #2ecc71;
    border-radius: 8px;
    padding: 20px 24px;
    margin-top: 16px;
}

.metric-row {
    display: flex;
    gap: 24px;
    margin-top: 12px;
}

.metric-box {
    background: rgba(255,255,255,0.04);
    border-radius: 6px;
    padding: 10px 16px;
    flex: 1;
    text-align: center;
}

.metric-label {
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 4px;
}

.metric-value {
    font-size: 22px;
    font-weight: 500;
    font-family: 'Syne', sans-serif;
}

.badge-drift   { color: #e74c3c; }
.badge-nodrift { color: #2ecc71; }
.badge-conf    { color: #f0c040; }

.sample-meta {
    font-size: 11px;
    color: #666;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}

.divider {
    border: none;
    border-top: 1px solid #222;
    margin: 24px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=2,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


# ── Dataset loader ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame | None:
    path = Path(DATASET_GLOB)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.dropna(subset=["window_text"])
    df = df[df["window_text"].str.len() >= 100]
    return df


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(text: str, tokenizer, model) -> dict:
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


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("# 📡 Topic Drift Detector")
st.markdown(
    "<span style='color:#666;font-size:13px;letter-spacing:0.1em'>"
    "LLAMA 3.2 · LORA FINE-TUNED · AMI/ICSI CORPUS"
    "</span>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# Load model
try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"Failed to load model from `{MODEL_PATH}`: {e}")
    st.stop()

# Load dataset (optional)
df = load_dataset()

# ── Input area ────────────────────────────────────────────────────────────────
col_label, col_btn = st.columns([3, 1])
with col_label:
    st.markdown("**Meeting transcript window**")
with col_btn:
    if df is not None:
        pull = st.button("🎲 Random sample", use_container_width=True)
    else:
        st.caption("No dataset found")
        pull = False

# ── Turn parser ───────────────────────────────────────────────────────────────
def parse_turns(raw: str) -> list[tuple[str, str]]:
    """
    Split a window_text string into (speaker, utterance) pairs.
    Handles AMI inline format: 'A: text B: more text C: ...'
    as well as newline-separated format.
    """
    import re
    # Flatten newlines into spaces so inline and multiline both work
    flat = " ".join(raw.strip().split("\n"))
    # Find all speaker token positions: 1-3 uppercase letters followed by ': '
    token_pat = re.compile(r'\b([A-Z]{1,3})\s*:\s*')
    matches = list(token_pat.finditer(flat))
    if not matches:
        return [("?", flat.strip())]
    turns = []
    for idx, m in enumerate(matches):
        spk = m.group(1)
        utt_start = m.end()
        utt_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(flat)
        utt = flat[utt_start:utt_end].strip()
        if utt:
            turns.append((spk, utt))
    return turns if turns else [("?", flat.strip())]


def turns_to_text(turns: list[tuple[str, str]]) -> str:
    return "\n".join(f"{spk}: {utt}" for spk, utt in turns)


# ── Default turns ─────────────────────────────────────────────────────────────
DEFAULT_TURNS = [
    ("A", "Okay so let's move on to the budget section."),
    ("B", "Right, we have about fifteen thousand left in Q3."),
    ("A", "And the projection for Q4 looks similar."),
    ("B", "Actually, did anyone watch the game last night? Incredible match."),
]
 
if "turns" not in st.session_state:
    st.session_state.turns = DEFAULT_TURNS
if "turn_version" not in st.session_state:
    st.session_state.turn_version = 0
 
 
# ── Random sample pull ────────────────────────────────────────────────────────
if pull and df is not None:
    row = df.sample(1).iloc[0]
    st.session_state.turns = parse_turns(row["window_text"])
    st.session_state.turn_version += 1
    st.session_state.sample_meta = {
        "session":  row.get("session_id", "—"),
        "agenda":   row.get("agenda_item", "—"),
        "label":    int(row.get("drift_label", -1)),
    }
    st.rerun()
 
# ── Per-turn inputs ───────────────────────────────────────────────────────────
if "sample_meta" in st.session_state:
    m = st.session_state.sample_meta
    label_str = "drift" if m["label"] == 1 else "no drift" if m["label"] == 0 else "unknown"
    st.markdown(
        f"<div class='sample-meta'>"
        f"SESSION: {m['session']} &nbsp;·&nbsp; AGENDA: {m['agenda']} &nbsp;·&nbsp; "
        f"TRUE LABEL: {label_str.upper()}"
        f"</div>",
        unsafe_allow_html=True,
    )
 
st.markdown("**Transcript turns**")
 
# Use version-stamped keys so Streamlit treats them as brand-new widgets
# whenever turns are replaced, forcing value= to take effect.
v = st.session_state.turn_version
edited_turns = []
for i, (spk, utt) in enumerate(st.session_state.turns):
    col_spk, col_utt = st.columns([1, 6])
    with col_spk:
        new_spk = st.text_input(
            label=f"spkr_{v}_{i}",
            label_visibility="collapsed",
            value=spk,
        )
    with col_utt:
        new_utt = st.text_input(
            label=f"utt_{v}_{i}",
            label_visibility="collapsed",
            value=utt,
        )
    edited_turns.append((new_spk.strip(), new_utt.strip()))
 
# Add / remove turn buttons
col_add, col_rem, col_run = st.columns([1, 1, 4])
with col_add:
    if st.button("＋ Add turn", use_container_width=True):
        st.session_state.turns = edited_turns + [("?", "")]
        st.session_state.turn_version += 1
        st.session_state.pop("sample_meta", None)
        st.rerun()
with col_rem:
    if st.button("－ Remove last", use_container_width=True) and len(edited_turns) > 1:
        st.session_state.turns = edited_turns[:-1]
        st.session_state.turn_version += 1
        st.session_state.pop("sample_meta", None)
        st.rerun()
with col_run:
    run = st.button("▶  Run inference", type="primary", use_container_width=True)
 
# Assemble final text from edited turns
text_input = turns_to_text(edited_turns)
 
# ── Result ────────────────────────────────────────────────────────────────────
if run:
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Running…"):
            result = predict(text_input.strip(), tokenizer, model)
 
        card_class = "drift-yes" if result["drift"] else "drift-no"
        verdict    = "⚠ DRIFT DETECTED" if result["drift"] else "✓ ON TOPIC"
        conf_pct   = f"{result['confidence']*100:.1f}%"
        nd_pct     = f"{result['no_drift_prob']*100:.1f}%"
        d_pct      = f"{result['drift_prob']*100:.1f}%"
        badge      = "badge-drift" if result["drift"] else "badge-nodrift"
 
        st.markdown(f"""
        <div class="{card_class}">
            <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:800;
                        letter-spacing:0.05em" class="{badge}">
                {verdict}
            </div>
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value badge-conf">{conf_pct}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">No Drift prob</div>
                    <div class="metric-value badge-nodrift">{nd_pct}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Drift prob</div>
                    <div class="metric-value badge-drift">{d_pct}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown(
    "<span style='color:#444;font-size:11px'>"
    f"Model: {MODEL_PATH} &nbsp;·&nbsp; Max seq len: {MAX_SEQ_LEN} &nbsp;·&nbsp; "
    f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}"
    "</span>",
    unsafe_allow_html=True,
)