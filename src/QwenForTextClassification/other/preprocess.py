import pandas as pd
from datasets import Dataset

df = pd.read_csv("./data.csv")


idx, _ = pd.factorize(df["intent"])

# label2idx = dict(zip(df["intent"], idx))
# print(label2idx)
# idx2label = {v: k for k, v in label2idx.items()}

# df["intent"] = df["intent"].map(label2idx)

df.to_json("./data.jsonl", orient="records", lines=True, force_ascii=False)
