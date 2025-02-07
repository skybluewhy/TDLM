from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import json
import random

with open("./bert_model/vocab.json", 'r', encoding='UTF-8') as f:
    token_dict = json.load(f)


class mydataset(Dataset):
    def __init__(self, filename, max_len=30):
        self.all_input_ids = []
        self.all_freq_ids = []
        self.all_attention_mask = []
        dat = pd.read_csv(filename, delimiter=",")
        for i in range(len(dat)):
            if i > 100:
                break
            line = dat["junction_aa"][i]
            if line[0] != "C" or line[-1] != 'F':
                continue
            input_id = []
            word_freq_id = []
            attention_mask = []
            l = 0
            while l < min(max_len, len(line)):
                input_id.append(token_dict[line[l]])
                word_freq_id.append(0)
                attention_mask.append(1)
                l += 1
            self.all_input_ids.append(input_id)
            self.all_freq_ids.append(word_freq_id)
            self.all_attention_mask.append(attention_mask)
        self.cnt = len(self.all_input_ids)

    def __len__(self):
        return self.cnt

    def __getitem__(self, index):
        return self.all_input_ids[index], self.all_attention_mask[index], self.all_freq_ids[index]


class mydataset_eval(Dataset):
    def __init__(self, filename, max_len=30):
        self.all_input_ids = []
        self.all_freq_ids = []
        self.all_attention_mask = []
        self.all_target_mask = []
        dat = pd.read_csv(filename, delimiter=",")
        for i in range(len(dat)):
            if i > 100:
                break
            line = dat["junction_aa"][i]
            input_id = []
            word_freq_id = []
            attention_mask = []
            target_mask = []
            l = 0
            while l < min(max_len, len(line)):
                input_id.append(token_dict[line[l]])
                word_freq_id.append(0)
                attention_mask.append(1)
                r = random.randint(0, 10)
                x = 1 if r <= 5 else 0
                target_mask.append(x)
                l += 1
            self.all_input_ids.append(input_id)
            self.all_freq_ids.append(word_freq_id)
            self.all_attention_mask.append(attention_mask)
            self.all_target_mask.append(target_mask)
        self.cnt = len(self.all_input_ids)

    def __len__(self):
        return self.cnt

    def __getitem__(self, index):
        return self.all_input_ids[index], self.all_attention_mask[index], self.all_target_mask[index]
