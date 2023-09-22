# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import pandas as pd

import torch
from torch.utils.data import Dataset


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class KBGenDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=200):
        data_path = dataset_config.data_path
        if partition == "train":
            # load jsonl
            self.data = pd.read_json(path_or_buf=data_path+"/train/train_llama.chunk.00.jsonl", lines=True)
        else:
            self.data = pd.read_json(path_or_buf=data_path+"/val/val_llama.chunk.00.jsonl", lines=True)

        self.data = self.data.values.flatten()

        self.max_words = max_words
        # tokenizer = Tokenizer(model_path=model_path + "./tokenizer.model")
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        assert isinstance(index, int)
        assert 0 <= index < len(self)

        prompt = self.data[index].replace("NaN", self.tokenizer.pad_token)

        output = self.tokenizer(prompt)

        output["input_ids"].append(self.tokenizer.eos_token_id)

        if len(output["input_ids"]) > self.max_words:
          # warn
          print(f"Warning: Truncating input to max_words, from {len(output['input_ids'])} to {self.max_words}")
          output["input_ids"] = output["input_ids"][:self.max_words]


        for _ in range(len(output["input_ids"]), self.max_words):
            output["input_ids"].append(self.tokenizer.pad_token_id)
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.int64)

        # 0 for pad token, 1 otherwise
        output["attention_mask"] = output["input_ids"] != self.tokenizer.pad_token_id

        output["labels"] = copy.deepcopy(output["input_ids"])
        output["labels"][output["labels"] == self.tokenizer.pad_token_id] = -100

        return output
