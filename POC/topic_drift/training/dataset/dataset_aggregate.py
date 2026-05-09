import glob
import pandas as pd

for split in ["train", "test", "validate"]:
    files = glob.glob(f"*_{split}.csv")
    if not files:
        print(f"No files found for: *_{split}.csv")
        continue

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out = f"aggregate_{split}.csv"
    df.to_csv(out)
    print(f"aggregate_{split}.csv <- {files} ({len(df)} rows)")