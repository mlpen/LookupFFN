
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

class MLMCollator:

    def __init__(
            self,
            tokenizer,
            num_masked_tokens,
            max_sequence_length,
        ):

        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = 1e9
        self.num_masked_tokens = num_masked_tokens
        self.max_sequence_length = max_sequence_length

    def mlm_masking(self, sequence, special_tokens_mask):
        tokenizer = self.tokenizer
        assert len(sequence.shape) == 2
        batch_size, sequence_length = sequence.shape

        inputs = sequence.clone()
        labels = sequence.clone()
        batch_indices = torch.arange(batch_size)[:, None]

        masked_token_noise = torch.rand(labels.shape) - 1000 * special_tokens_mask.float()
        masked_token_ranking = torch.argsort(masked_token_noise, descending = True, dim = -1)
        masked_token_mask = torch.zeros_like(masked_token_ranking)
        masked_token_mask[batch_indices, masked_token_ranking[:, :self.num_masked_tokens]] = 1

        masked_token_indices = torch.sort(masked_token_mask, descending = True, stable = True, dim = -1).indices
        masked_token_indices = masked_token_indices[:, :self.num_masked_tokens]

        masked_token_mask = masked_token_mask.bool()

        labels[~masked_token_mask] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_token_mask
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_token_mask & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype = torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        return inputs, labels, masked_token_indices

    def process_one_sequence(self, sequence):

        tokenizer = self.tokenizer

        sequence = sequence.replace(tokenizer.eos_token, "")
        sequence = tokenizer.encode(sequence, add_special_tokens = False)

        if len(sequence) - (self.max_sequence_length - 2) > 0:
            truncate_start = random.choice(range(len(sequence) - (self.max_sequence_length - 2)))
        else:
            truncate_start = 0
        truncate_end = truncate_start + (self.max_sequence_length - 2)
        sequence = [tokenizer.convert_tokens_to_ids(tokenizer.cls_token)] + sequence[truncate_start:truncate_end] + [tokenizer.convert_tokens_to_ids(tokenizer.sep_token)]

        segment_ids = [0] * len(sequence)
        sequence_mask = [1] * len(sequence)
        pos_ids = list(range(len(sequence)))

        while len(sequence) < self.max_sequence_length:
            sequence.append(tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
            pos_ids.append(0)
            segment_ids.append(0)
            sequence_mask.append(0)

        assert len(sequence) == self.max_sequence_length

        special_tokens_mask = tokenizer.get_special_tokens_mask(sequence, already_has_special_tokens = True)

        sequence = torch.tensor(sequence, dtype = torch.long)
        pos_ids = torch.tensor(pos_ids, dtype = torch.long)
        segment_ids = torch.tensor(segment_ids, dtype = torch.long)
        sequence_mask = torch.tensor(sequence_mask, dtype = torch.long)
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype = torch.long)

        instance = {
            "sequence":sequence,
            "special_tokens_mask":special_tokens_mask,
            "position_ids":pos_ids,
            "token_type_ids":segment_ids,
            "attention_mask":sequence_mask,
        }

        return instance

    def __call__(self, list_strs):
        # list_strs is a list of batch_size strings

        batch = {}
        for sequence in list_strs:
            instance = self.process_one_sequence(sequence)
            for key in instance:
                if key not in batch:
                    batch[key] = []
                batch[key].append(instance[key])

        batch = {key:torch.stack(batch[key], dim = 0) for key in batch}

        mlm_sequence, mlm_labels, masked_token_indices = self.mlm_masking(batch["sequence"], batch["special_tokens_mask"])
        del batch["sequence"]
        del batch["special_tokens_mask"]
        batch["importance_mask"] = (mlm_labels != -100).long()
        batch["masked_token_indices"] = masked_token_indices
        batch["input_ids"] = mlm_sequence
        batch["labels"] = mlm_labels

        return batch
