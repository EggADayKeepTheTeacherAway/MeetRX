import requests
import pandas as pd

urls = {"test": "https://huggingface.co/datasets/Coldog2333/tiage/resolve/main/test.json?download=true",
        "train": "https://huggingface.co/datasets/Coldog2333/tiage/resolve/main/train.json?download=true",
        "validate": "https://huggingface.co/datasets/Coldog2333/tiage/resolve/main/validation.json?download=true"}


def load_data(name, url):
    r = requests.get(url)
    raw = r.json()

    # print(str(raw["dial_data"])[:100])
    rows = []
    for dialog in raw["dial_data"]['tiage']:
        # print(dialog)
        dial_id = dialog["dial_id"]
        for turn in dialog["turns"]:
            rows.append({
                "dial_id": dial_id,
                "turn_id": turn["turn_id"],
                "role": turn["role"],
                "utterance": turn["utterance"],
                "topic_id": turn["topic_id"],
                "segmentation_label": turn["segmentation_label"],
                "da": turn["da"],
            })

    df = pd.DataFrame(rows)
    print(df.head())
    print(df.shape)

    df.to_csv(f"tiage_{name}.csv")

    # df = pd.DataFrame(raw["dial_data"])
    # print(df.head())

    # print(raw.values())

for name, url in urls.items():
    load_data(name, url)