

import json
from datasets import Dataset
import pandas as pd

def create_dataset(path):
    with open(path, "r") as f:
        lines = f.readlines()
    print(len(lines))
    dataset = []
    for line in lines:
        data = json.loads(line)
        context = data["article"]
        for question in data["questions"]:
            instance = {"context":context, "question":question["question"], "options":question["options"],}
            if "gold_label" in question:
                instance["answer"] = question["gold_label"] - 1
            dataset.append(instance)
    dataset = Dataset.from_pandas(pd.DataFrame(dataset))
    return dataset


train = create_dataset("src/datasets_scripts/downstream/data_input/quality/QuALITY.v1.0.1.htmlstripped.train")
dev = create_dataset("src/datasets_scripts/downstream/data_input/quality/QuALITY.v1.0.1.htmlstripped.dev")
test = create_dataset("src/datasets_scripts/downstream/data_input/quality/QuALITY.v1.0.1.htmlstripped.test")

print(len(train), len(dev), len(test))

train.save_to_disk("src/datasets_scripts/downstream/data_input/datasets/quality/train")
dev.save_to_disk("src/datasets_scripts/downstream/data_input/datasets/quality/dev")
test.save_to_disk("src/datasets_scripts/downstream/data_input/datasets/quality/test")
