import pandas as pd

WINDOW_SIZE = 125  # words

def extract_windows(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for dial_id, group in df.groupby("dial_id", sort=False):
        group = group.sort_values("turn_id").reset_index(drop=True)

        window_text = []
        window_turns = []
        word_count = 0

        def flush_window():
            if not window_turns:
                return
            text = " ".join(window_text)
            label = int(any(r["segmentation_label"] == 1 for r in window_turns))
            topic_ids = list(dict.fromkeys(r["topic_id"] for r in window_turns))
            records.append({
                "dial_id": dial_id,
                "turn_start": window_turns[0]["turn_id"],
                "turn_end": window_turns[-1]["turn_id"],
                "topic_ids": topic_ids,
                "word_count": word_count,
                "window_text": text,
                "drift_label": label,
            })

        for _, row in group.iterrows():
            words = str(row["utterance"]).split()

            # If adding this turn exceeds the window, flush first
            if word_count + len(words) > WINDOW_SIZE and window_turns:
                flush_window()
                window_text = []
                window_turns = []
                word_count = 0

            window_text.append(str(row["utterance"]))
            window_turns.append(row)
            word_count += len(words)

        # Flush remaining turns at end of dialogue
        flush_window()

    return pd.DataFrame(records)


if __name__ == "__main__":
    import sys

    input_file = sys.argv[1] if len(sys.argv) > 1 else "csvfile.csv"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "windows.csv"

    df = pd.read_csv(input_file)
    df["segmentation_label"] = pd.to_numeric(df["segmentation_label"], errors="coerce").fillna(0).astype(int)

    result = extract_windows(df)
    result.to_csv(output_file, index=False)

    total = len(result)
    drifted = result["drift_label"].sum()
    print(f"Input    : {input_file} ({len(df)} rows)")
    print(f"Output   : {output_file} ({total} windows)")
    print(f"Drifted  : {drifted} ({drifted/total*100:.1f}%)")
    print(f"No drift : {total - drifted} ({(total-drifted)/total*100:.1f}%)")
    print()
    print(result.to_string(index=False))