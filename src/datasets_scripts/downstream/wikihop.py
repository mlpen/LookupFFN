import json
from datasets import Dataset
import pandas as pd

with open("src/datasets_scripts/downstream/data_input/qangaroo_v1.1/wikihop/dev.json") as f:
    dev = json.load(f)
with open("src/datasets_scripts/downstream/data_input/qangaroo_v1.1/wikihop/train.json") as f:
    train = json.load(f)

dev = Dataset.from_pandas(pd.DataFrame(dev))
train = Dataset.from_pandas(pd.DataFrame(train))

print(len(dev), len(train))

dev.save_to_disk("src/datasets_scripts/downstream/arrow_output/wikihop/dev")
train.save_to_disk("src/datasets_scripts/downstream/arrow_output/wikihop/train")
