import requests
import pandas as pd

url = "https://huggingface.co/datasets/Coldog2333/dialseg711/resolve/main/test.json"
r = requests.get(url)
raw = r.json()


rows = []
for dialog in raw["dial_data"]['dialseg711']:
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

df.to_csv("dialseg711.csv")

from sklearn.model_selection import train_test_split

df = pd.read_csv("dialseg711.csv")

# First split off 10% for validation
train_temp, val = train_test_split(df, test_size=0.10, random_state=42)

# Then split remaining 90% into 70% train / 20% test (20/90 ≈ 0.222)
train, test = train_test_split(train_temp, test_size=0.2222, random_state=42)

train.to_csv("dialseg_train.csv")
test.to_csv("dialseg_test.csv")
val.to_csv("dialseg_validate.csv")

print(f"Total  : {len(df)}")
print(f"Train  : {len(train)} ({len(train)/len(df)*100:.1f}%)")
print(f"Test   : {len(test)} ({len(test)/len(df)*100:.1f}%)")
print(f"Val    : {len(val)} ({len(val)/len(df)*100:.1f}%)")
