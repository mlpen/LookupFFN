
import datasets
import argparse
import os
import multiprocessing
from multiprocessing import Pool
import numpy as np
import pyarrow as pa
import json
import pyarrow.dataset as ds
import time
from tqdm import tqdm
from transformers import AutoTokenizer
from itertools import repeat
from collections import deque
import warnings
import pandas as pd
import os
import spacy
import argparse

def segment_sentence_example_pack(doc_list, example_pack_length):
    nlp = spacy.load("en_core_web_sm")
    eos = tokenizer.encode(tokenizer.eos_token, add_special_tokens = False)[0]
    packed_examples = []
    example = []
    example_length = 0
    for doc_idx, doc in enumerate(doc_list):
        if len(doc) > nlp.max_length:
            len_doc = len(doc)
            segments = [doc[breakpoint:(breakpoint + nlp.max_length)] for breakpoint in range(0, len_doc, nlp.max_length)]
            print(f"Splitting {len_doc}-character document to {len(segments)} {nlp.max_length}-character segments")
        else:
            segments = [doc]

        for segment_idx, segment in enumerate(segments):
            segment = nlp(segment)
            for sent in segment.sents:
                sent = tokenizer.encode(sent.text, add_special_tokens = False)
                if len(sent) == 0:
                    continue
                example.extend(sent)
                example.append(eos)
                example_length += len(sent)
                if example_length >= example_pack_length:
                    packed_examples.append(tokenizer.decode(example))
                    example = []
                    example_length = 0

        example.append(eos)
        example.append(eos)

    return packed_examples

def example_pack(doc_list, example_pack_length):
    packed_examples = []
    max_length = tokenizer.model_max_length
    eos = tokenizer.encode(tokenizer.eos_token, add_special_tokens = False)[0]

    tokens = []
    for doc in doc_list:
        tokens.extend(tokenizer.encode(doc, add_special_tokens = False))
        tokens.extend([eos, eos])

    packed_examples = []
    for breakpoint in range(0, len(tokens), example_pack_length):
        example = tokens[breakpoint:(breakpoint + example_pack_length)]
        if len(example) < example_pack_length:
            continue
        packed_examples.append(tokenizer.decode(example))

    return packed_examples

def example_pack_from_dataset(dataset, offset, work_size, tokenizer, example_pack_length, output_folder, segment_sentences):
    work_size = min(work_size, len(dataset) - offset)
    print(f"{time.time()}, Working from {offset} to {offset + work_size}, total {len(dataset)}")
    doc_list = dataset.select(range(offset, offset + work_size))
    doc_list = [x['text'] for x in doc_list]
    packed_examples = segment_sentence_example_pack(doc_list, example_pack_length) if segment_sentences else example_pack(doc_list, example_pack_length)
    with open(os.path.join(output_folder, f"{offset:010d}-to-{(offset + work_size):010d}.jsonl"), "w") as f:
        for inst in packed_examples:
            f.write(json.dumps({"text":inst}))
            f.write("\n")

    log = [
        f"{time.time()}",
        f"Offset {offset}, Size {len(dataset)}",
        f"Number of documents {len(doc_list)}",
        f"Number of packed examples {len(packed_examples)}",
    ]
    print(", ".join(log))
    return len(packed_examples)


def example_pack_from_jsonl(data_folder, file, tokenizer, example_pack_length, output_folder, segment_sentences):
    if os.path.exists(os.path.join(output_folder, file)):
        print(f"File {os.path.join(output_folder, file)} exists, skipping")
        try:
            with open(os.path.join(output_folder, file), "r") as f:
                lines = [json.loads(line) for line in f.readlines()]
            return len(lines)
        except Exception as e:
            print(e)

    print(f"Processing {os.path.join(data_folder, file)}")
    with open(os.path.join(data_folder, file), "r") as f:
        lines = f.readlines()
    doc_list = [json.loads(line) for line in lines]
    doc_list = [x['text'] for x in doc_list]
    packed_examples = segment_sentence_example_pack(doc_list, example_pack_length) if segment_sentences else example_pack(doc_list, example_pack_length)

    with open(os.path.join(output_folder, file), "w") as f:
        for example in packed_examples:
            f.write(json.dumps({"text":example}) + "\n")

    log = [
        f"{time.time()}",
        f"Process {os.path.join(data_folder, file)}",
        f"Output {os.path.join(output_folder, file)}",
        f"Number of documents {len(doc_list)}",
        f"Number of packed examples {len(packed_examples)}",
    ]
    print(", ".join(log))
    return len(packed_examples)

os.environ['TOKENIZERS_PARALLELISM'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument("--output_folder", type = str, required = True)
parser.add_argument("--data_file", type = str, default = None)
parser.add_argument("--data_folder", type = str, default = None)
parser.add_argument("--work_size", type = int, default = 5000)
parser.add_argument("--tokenizer", type = str, default = 'roberta-base')
parser.add_argument("--example_pack_length", type = int, default = 4096)
parser.add_argument("--segment_sentences", action = 'store_true')
parser.add_argument("--mp", action = 'store_true')
args = parser.parse_args()

print(args)

os.makedirs(args.output_folder, exist_ok = True)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
tokenizer.model_max_length = 2147483647
num_cores = os.cpu_count()
print("num_cores", num_cores)

if args.data_file is not None:
    dataset = datasets.load_from_disk(args.data_file).shuffle(seed = 1)
    offsets = list(range(0, len(dataset), args.work_size))
    print("number of tasks", len(offsets))
    if args.mp:
        with Pool(processes = num_cores) as pool:
            results = [pool.apply_async(example_pack_from_dataset, (dataset, offset, args.work_size, tokenizer, args.example_pack_length, args.output_folder, args.segment_sentences)) for offset in offsets]
            total_examples = sum([res.get() for res in results])
    else:
        results = [example_pack_from_dataset(dataset, offset, args.work_size, tokenizer, args.example_pack_length, args.output_folder, args.segment_sentences) for offset in offsets]
        total_examples = sum(results)
elif args.data_folder is not None:
    train_files = [os.path.join("train", file) for file in os.listdir(os.path.join(args.data_folder, "train")) if file.endswith(".jsonl")]
    val_files = [os.path.join("val", file) for file in os.listdir(os.path.join(args.data_folder, "val")) if file.endswith(".jsonl")]
    test_files = [os.path.join("test", file) for file in os.listdir(os.path.join(args.data_folder, "test")) if file.endswith(".jsonl")]
    os.makedirs(os.path.join(args.output_folder, "train"), exist_ok = True)
    os.makedirs(os.path.join(args.output_folder, "val"), exist_ok = True)
    os.makedirs(os.path.join(args.output_folder, "test"), exist_ok = True)
    print("train_files", len(train_files))
    print("val_files", len(val_files))
    print("test_files", len(test_files))
    files = val_files + test_files + train_files
    print("number of tasks", len(files))
    if args.mp:
        with Pool(processes = num_cores) as pool:
            results = [pool.apply_async(example_pack_from_jsonl, (args.data_folder, file, tokenizer, args.example_pack_length, args.output_folder, args.segment_sentences)) for file in files]
            total_examples = sum([res.get() for res in results])
    else:
        results = [example_pack_from_jsonl(args.data_folder, file, tokenizer, args.example_pack_length, args.output_folder, args.segment_sentences) for file in files]
        total_examples = sum(results)

print("total_examples", total_examples)
