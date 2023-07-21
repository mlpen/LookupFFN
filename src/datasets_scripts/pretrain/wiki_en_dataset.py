from datasets import load_dataset
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--output", type = str, required = True)
args = parser.parse_args()

ds = load_dataset("wikipedia", "20220301.en")
ds.save_to_disk(os.path.join(args.output, "wikipedia.20220301.en"))
