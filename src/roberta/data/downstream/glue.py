
import torch
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
import numpy as np
from transformers import PreTrainedTokenizerBase
from transformers import AutoTokenizer
import random
import numpy as np
import torch

class MNLICollator:

    def __init__(
            self,
            tokenizer,
            max_sequence_length,
            encode_type,
        ):

        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = 1e9
        self.max_sequence_length = max_sequence_length
        self.encode_type = encode_type

    def process_one_instance(self, inst):

        tokenizer = self.tokenizer

        sentence_A = tokenizer.encode(inst["premise"])
        sentence_B = tokenizer.encode(inst["hypothesis"])

        if self.encode_type == "original":
            sequence = sentence_A + sentence_B[1:]
            segment_ids = [0] * len(sequence)
        elif self.encode_type == "gsop":
            sequence = sentence_A + sentence_B
            segment_ids = [0] * len(sentence_A) + [1] * len(sentence_B)
        else:
            raise Exception()

        sequence_mask = [1] * len(sequence)
        pos_ids = list(range(len(sequence)))

        # truncate or pad sequence to max_sequence_length
        if len(sequence) > self.max_sequence_length:
            sequence = sequence[:self.max_sequence_length]
            pos_ids = pos_ids[:self.max_sequence_length]
            segment_ids = segment_ids[:self.max_sequence_length]
            sequence_mask = sequence_mask[:self.max_sequence_length]

        while len(sequence) < self.max_sequence_length:
            sequence.append(tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
            pos_ids.append(0)
            segment_ids.append(0)
            sequence_mask.append(0)

        sequence = torch.tensor(sequence, dtype = torch.long)
        pos_ids = torch.tensor(pos_ids, dtype = torch.long)
        segment_ids = torch.tensor(segment_ids, dtype = torch.long)
        sequence_mask = torch.tensor(sequence_mask, dtype = torch.long)

        instance = {
            "input_ids":sequence,
            "position_ids":pos_ids,
            "token_type_ids":segment_ids,
            "attention_mask":sequence_mask,
            "label":torch.tensor(inst["label"], dtype = torch.long),
        }

        return instance

    def __call__(self, instances):
        batch = {}
        for inst in instances:
            inst = self.process_one_instance(inst)
            for key in inst:
                if key not in batch:
                    batch[key] = []
                batch[key].append(inst[key])

        batch = {key:torch.stack(batch[key], dim = 0) for key in batch}

        return batch
