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
base_path = '/misc/vlgscratch2/LecunGroup/anant/nlp'
use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.cuda.set_device(3)

batch_size = 64
num_workers = 1
hidden_dim = 400
random.seed(10)


# In[42]:

def splitdata(rawdata):
    random.shuffle(rawdata)
    margin = int(len(rawdata)*0.8)
    traindata = rawdata[:margin]
    valdata = rawdata[margin:]
    return (traindata, valdata)

def read_data_dump(data_path, token2idx):
    with open(data_path, 'r') as f:
        rawdata = pickle.load(f)['data']

    for i, key in enumerate(rawdata):
        rawdata[i]['notes'] = [[token2idx.get(token, token2idx['unknown']) for token in note.split(' ')] for note in key['notes']][-5:]
    return splitdata(rawdata)

def read_data_dump_v2(data_path, token2idx):
    #    Drop all unknowns and keep notes with len > 50
    with open(data_path, 'r') as f:
        rawdata = pickle.load(f)['data']
    newrawdata = []
    for i, key in enumerate(rawdata):
        token_notes = []
        for note in key['notes']:
            note_idx = []
            for token in note.split(' '):
                if (token2idx.get(token,None) is not None and token!='unknown'):
                    note_idx.append(token2idx[token])
            if len(note_idx) > 50: 
                token_notes.append(note_idx)
        rawdata[i]['notes'] = token_notes
        if len(token_notes):
            newrawdata.append(rawdata[i])
    rawdata = newrawdata
    return splitdata(rawdata)

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



def read_embeddings(vec_path):    
    vecs, tokens = [], []
    with open(vec_path, 'r') as f:
        tsvreader = csv.reader(f, delimiter='\t')
        for i,row in enumerate(tsvreader):
            tokens.append(row[0])
            vecs.append(np.array([float(_) for _ in row[1:]]).reshape(1, -1))
    vecs = np.concatenate(vecs)
    token2idx = {token:_ for _, token in enumerate(tokens)}
    return (vecs, token2idx)

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


class NoteDataloader(Dataset):
    def __init__(self, data):
        super(Dataloader, self).__init__()
        self.data = data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)
    
def padding_collation(batch):
    batch_x, batch_y = [], []
    max_seq_len = np.max([len(key['notes']) for key in batch])
    
    for key in range(len(batch)):
        x = batch[key]['notes']
        note_max_seq_len = np.max([len(note) for note in x])
        #x= [[1759 for i in range(note_max_seq_len-len(note))]+note for note in x]

        # NOTE hardcoded value of 'pad'. Automate this!! Make padding 0 next time
        x = [[1759] for i in range(max_seq_len-len(x))] + x
        #x = [np.array(_) for _ in x]
        y =  batch[key]['labels']['icd'][0]
        batch_x.append(x)
        batch_y.append(y)
    return (batch_x, batch_y)


class LSTMModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_labels, batch_size, vocab_size, pretrained=None):
        super(LSTMModel, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(vocab_size, embed_dim)
        #self.embed.weight = nn.Parameter(torch.from_numpy(pretrained).float())

        #Attention!!!
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=False)
        self.lin = nn.Linear(hidden_dim, num_labels)

    def init_hidden(self):
        hidden1 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        hidden2 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        if use_cuda:
            return (hidden1.cuda(), hidden2.cuda())
        else:
            return (hidden1, hidden2)
    def forward(self, x, hidden):
        # seqlen x batch x emb_dim

        # batch x seq_len x list of idx  -> batch x seq_len x embed_size
        #avg_embeds = []
        #for b in range(batch_size):
        #    avg_embeds.append(torch.mean(embed(x[b, :, :]), dim=1))
        #x = torch.cat(avg_embeds, dim=0)
        x = self.embed(x)
        x = torch.transpose(x, 1, 0)
        x, _hidden  = self.lstm(x, hidden)
        x = x[-1, :, :].view(self.batch_size, -1)
        x = F.sigmoid(self.lin(x))
        return x


# In[144]:

def data_iter(source, batch_size):
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while True:
        start += batch_size
        if start > dataset_size - batch_size:
            # Start another epoch.
            start = 0
            random.shuffle(order)   
        batch_indices = order[start:start + batch_size]
        yield [source[index] for index in batch_indices]

if __name__ == "__main__":
    print "Reading data..."
    lr = 1e-2
    num_epochs = 10000
    _time = time.time()
    pretrained, token2idx = read_embeddings(os.path.join(base_path, 'ri-3gram-400-tsv/filtered_embeddings.tsv'))
    traindata, valdata = read_data_dump(os.path.join(base_path, 'notes_dump_cleaned.pkl'), token2idx)
    label_map = get_labels(os.path.join(base_path, 'labels.txt'))
    print "Preprocessing done in %.2f"%((time.time()-_time)/60)


    model = LSTMModel(400, hidden_dim, len(label_map.keys()), batch_size, pretrained)

    opti = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    model.embed.weight.requires_grad = False
    crit = nn.CrossEntropyLoss()

    if use_cuda:
        model.cuda()
        crit.cuda()

    #train_loader = torch.utils.data.DataLoader(dataset=Dataloader(traindata), batch_size=batch_size, shuffle=True, 
    #                                                           num_workers=num_workers)
    #val_loader = torch.utils.data.DataLoader(dataset=Dataloader(valdata), batch_size=batch_size, shuffle=False) 


    training_iter = data_iter(traindata, batch_size)
    # In[147]:
    # In[ ]:

    step = 0
    step_log = []
    loss_log = []
    val_acc_log = []
    train_acc_log = []
    train_acc = []
    torch.manual_seed(1)


    for step in range(num_epochs):

        model.zero_grad()
        dp = next(training_iter)
        max_note_seq_len = max([len(_['notes']) for _ in dp])

        padded_batch_x = []
        for dp_num, note_seq in enumerate(dp):
            mnl =  np.max([len(_) for _ in note_seq['notes']])
            padded_notes=[]
            for _ in note_seq['notes']:
                padded_notes.append([0]*(mnl-len(_))+_)

            batch_x = Variable(torch.from_numpy(np.array(padded_notes)).long())
            if use_cuda:
                batch_x = batch_x.cuda()
            batch_x = model.embed(batch_x)
            batch_x = torch.mean(batch_x, dim=1).unsqueeze(0)

            if max_note_seq_len-batch_x.size(1):
                pads = torch.zeros((max_note_seq_len-batch_x.size(1), 400)).unsqueeze(0)
                pads = Variable(pads.float())
                if use_cuda:
                    pads = pads.cuda()

                batch_x = torch.cat([pads, batch_x], dim=1)
            padded_batch_x.append(batch_x)            
        batch_x = torch.cat(padded_batch_x, dim=0)
        labels = np.array([label_map[note_seq['labels']['icd'][0]] for note_seq in dp])
        batch_y = Variable(torch.from_numpy(labels).long())
        if use_cuda:
            batch_y =batch_y.cuda()

        hidden = model.init_hidden()
        x = model(batch_x, hidden)
        loss = crit(x, batch_y)
        loss.backward()
        opti.step()

        _, predicted = torch.max(x.data, 1)
        total = batch_y.size(0)
        correct = (predicted == batch_y.data).sum()
        train_acc.append(correct / float(total))
        
        if step % 100 == 0:
            '''
            val_acc = evaluate(model, val_loader, batch_size, label_map)
            print("Step: %d, Loss: %.4f, Train Acc: %.2f, Validation Acc: %.2f"%(step, loss.data[0], train_acc, val_acc))
            val_acc_log.append(val_acc)
            '''
            model.eval()
            train_acc = np.mean(train_acc)
            print("Step: %d, Loss: %.4f, Train Acc: %.2f"%(step, loss.data[0], train_acc))
            step_log.append(step)
            loss_log.append(loss.data[0])
            train_acc_log.append(train_acc)
            model.train()
            train_acc = []
            lr *= 0.9
            print "new LR:", lr
            for param_group in opti.param_groups:
                param_group['lr'] = lr
            
        step += 1
        #if step%10000==0:    
    f = open('results.pkl', 'w')
    pickle.dump({'step': step_log, 'loss': loss_log, 'train': train_acc_log}, f)
    #pickle.dump({'step': step_log, 'loss': loss_log, 'val': val_acc_log, 'train': train_acc_log}, f)
    f.close() 




