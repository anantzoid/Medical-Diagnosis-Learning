import csv
import re
import random
from collections import Counter
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import math

### NOTE this is temp
def get_labels(data):
    labels = []
    for row in data:
        labels.append(row[-1].split(' ')[0])
    return list(set(labels))

def build_vocab(data, PAD):
    vocab = []
    for row in data:
        for text in row[1]:
            vocab.extend(text)
    vocab = [PAD] + list(set(vocab))
    token2idx = {i:_ for _,i in enumerate(vocab)}
    return (vocab, token2idx)

class NotesData(Dataset):
    def __init__(self, data, token2idx,  UNKNOWN, label_map):
        super(NotesData, self).__init__()
        data_proc = []
        for i, row in enumerate(data):
            token_seq = [[token2idx.get(word, token2idx[UNKNOWN]) for word in sent] for sent in row[1]]
            label = label_map[row[2].split(' ')[0]]
            data_proc.append([token_seq, label])
        self.data = data_proc

    def __getitem__(self, index):
        return (self.data[index])
    def __len__(self):
        return len(self.data)

def sent_batch_collate(batch):
    max_note_len = max([len(_[0]) for _ in batch])
    max_sentence_len = max([len(i) for _ in batch for i in _[0]])
    
    x = torch.zeros(len(batch), max_note_len, max_sentence_len)
    for n,note in enumerate(batch):
        for s,sentence in enumerate(note[0]):
            for w, word in enumerate(sentence):                                
                x[n, s, w] = float(word)

    return (x.long(), torch.from_numpy(np.array([_[1] for _ in batch])).long())



def calc_grad_norm(parameters, norm_type):
    norm_type = float(norm_type)
    total_norm, total_grad_norm = 0, 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_grad_norm += param_norm ** norm_type

        param_norm = p.data.norm(norm_type)
        total_norm += param_norm ** norm_type

    total_norm = total_norm ** (1. / norm_type)
    total_grad_norm = total_grad_norm ** (1. / norm_type)
    return (total_norm, total_grad_norm)


