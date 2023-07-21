from datasets import load_dataset
import os
import argparse
import json
import os
import pyarrow as pa
import json
import numpy as np

def write_arrow_file(all_text, output_file):
    '''
    write all read and processed text into an arrow file
    '''
    try:
        schema = pa.schema({'text': pa.large_string()})
        arr = pa.array(all_text, type = pa.large_string())
        with pa.OSFile(output_file, 'wb') as sink:
            with pa.ipc.new_stream(sink, schema = schema) as writer:
                batch = pa.record_batch([arr], schema = schema)
                writer.write(batch)
        print("Finished writing {}".format(output_file))

    except Exception as e:
        print(e)
        return 0
    return len(arr)

parser = argparse.ArgumentParser()
parser.add_argument("--jsonl_folder", type = str, required = True)
parser.add_argument("--output_file", type = str, required = True)
args = parser.parse_args()

seed = 1234

all_text = []

files = sorted([file for file in os.listdir(args.jsonl_folder) if file.endswith(".jsonl")])

for file_idx, file in enumerate(files):
    file = os.path.join(args.jsonl_folder, file)
    print(file_idx, len(files), len(all_text), file)

    with open(file, "r") as f:
        lines = f.readlines()
    for line in lines:
        if len(line.strip()) == 0:
            continue
        line = json.loads(line)
        all_text.append(line["text"])

print()
print(len(all_text))

print(all_text[0][:100])

rng = np.random.default_rng(seed)
rng.shuffle(all_text)
print(len(all_text))
print(all_text[0][:100])

write_arrow_file(all_text, args.output_file)
