# coding: utf-8
import os
import pickle
import csv
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import re
import time
import random


# In[42]:

def splitdata(rawdata):
    random.shuffle(rawdata)
    margin = int(len(rawdata)*0.8)
    traindata = rawdata[:margin]
    valdata = rawdata[margin:]
    return (traindata, valdata)


def csv_prepare_sentence_wise(data_path, dest_path, dis_sum=False):
    with open(data_path, 'r') as f:
        rawdata = pickle.load(f)['data']
    newrawdata = []
    rawdata = rawdata
    for i, key in enumerate(rawdata):
        for note in key['notes']:
            if dis_sum == True: 
                if note['note_type'].lower() == 'discharge summary':
                    newrawdata.append([note['note'], key['labels']['icd'][0]])
            else:
                newrawdata.append([note['note'], key['labels']['icd'][0]])
    with open(dest_path, 'w') as csvf:
        writer = csv.writer(csvf, delimiter=',', quotechar='"')
        writer.writerows(newrawdata)
    
       
def read_data_dump_v3(data_path, token2idx=None):
    #    Keep last discharge summary    
    with open(data_path, 'r') as f:
        rawdata = pickle.load(f)['data']
    newrawdata = []
    for i, key in enumerate(rawdata):
        token_notes = []
        key['notes'].reverse()
        for note in key['notes']:
            if note['note_type'].lower() != 'discharge summary':
                continue
            note_idx = []
            note = note['note']
            for token in note.split(' '):
                if len(token) > 2:
                    note_idx.append(token)
                #if (token2idx.get(token,None) is not None and token!='unknown'):
                #    note_idx.append(token2idx[token])
            token_notes = note_idx
            break
        rawdata[i]['original'] = note
        rawdata[i]['notes'] = token_notes
        if len(token_notes):
            newrawdata.append(rawdata[i])
    rawdata = newrawdata
    return splitdata(rawdata)



#NOTE to be used later when icd9 labels are too large and
# needs to be mapped to ccs
def get_labels_ccs(ccs_path):
    with open(ccs_path, 'rb') as csvf:
        ccs_mapping = {}
        reader = csv.reader(csvf, delimiter=',')
        _ = next(reader)
        for i, row in enumerate(reader):
            ccs_mapping[row[0]] = row[1]
    ccs2idx = {code: i for i, code in enumerate(list(set([ccs_maping[k] for k in ccs_maping.keys()])))}
    icd2idx = {icd: ccs2idx[ccs_maping[icd]] for icd in ccs_maping.keys()}
    return icd2idx

def get_labels(labels_path):
    f = open(labels_path, 'r')
    labels = []
    for line in f.readlines():
        labels.append(line.replace('\n', ''))
    labels = {i:_ for _,i in enumerate(labels)}
    return labels

def gett2i(corpus):
    vocab_list = [] 
    for note in corpus:
        for t in note['notes']:
            vocab_list.append(t)
    return vocab_list

class NoteDataloader(Dataset):
    def __init__(self, data):
        super(NoteDataloader, self).__init__()
        self.data = data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)
 
class LSTMModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_labels, batch_size, vocab_size, use_cuda, pretrained=None):
        super(LSTMModel, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda

        self.embed = nn.Embedding(vocab_size, embed_dim)
        #self.embed.weight = nn.Parameter(torch.from_numpy(pretrained).float())

        #Attention!!!
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=False)
        self.lin = nn.Linear(hidden_dim, num_labels)

    def init_hidden(self):
        hidden1 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        hidden2 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        if self.use_cuda:
            return (hidden1.cuda(), hidden2.cuda())
        else:
            return (hidden1, hidden2)
    def forward(self, x, hidden):
        # seqlen x batch x emb_dim
        x = self.embed(x)
        x = torch.transpose(x, 1, 0)
        x, _hidden  = self.lstm(x, hidden)
        x = x[-1, :, :].view(self.batch_size, -1)
        x = F.sigmoid(self.lin(x))
        return x



if __name__ == "__main__":
    base_path = '/misc/vlgscratch2/LecunGroup/anant/nlp'
    csv_prepare_sentence_wise(os.path.join(base_path, 'notes_dump_cleaned.pkl'), os.path.join(base_path, 'notes_csv_5label.csv'))
