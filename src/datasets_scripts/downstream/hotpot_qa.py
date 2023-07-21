from datasets import load_dataset

dataset = load_dataset('hotpot_qa', 'distractor')
print(dataset)
dataset["train"].save_to_disk(f"src/datasets_scripts/downstream/arrow_output/hotpot_qa-distractor/train")
dataset["validation"].save_to_disk(f"src/datasets_scripts/downstream/arrow_output/hotpot_qa-distractor/validation")
