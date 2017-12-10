import os
import csv
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# will different hadm_id have same sequence_num continuation
#   find all subject id and confirm that
def build_encounter_seq(encounter_data_path, skip_read=False):
    patients = {}
    pat_adm_id = {}
    with open(encounter_data_path, 'rb') as csvf:
        csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
        skip_header = next(csvreader)
        for row in csvreader:
            if skip_read and len(patients.keys()) > 10:
                break
            if patients.get(row[1]) is None:
                patients[row[1]] = {row[2]: [[row[3], row[4]]]}
                pat_adm_id[row[1]] = [row[2]]
            elif patients[row[1]].get(row[2]) is None:
                patients[row[1]][row[2]] = [[row[3], row[4]]]
                pat_adm_id[row[1]].append(row[2])
            else:
                patients[row[1]][row[2]].append([row[3], row[4]])
    for key in patients:
        patients[key] = {_[0]: _[1]  for _ in sorted(patients[key].iteritems())} 
    return (patients, pat_adm_id)

def POC_of_seq(patients, pat_adm_id):
    # POC of a subject with multiple hadm_id and check seq_num
    for key, val in pat_adm_id.iteritems():
        if len(val) > 1:
            for v in val:
                print(key, v, patients[(key, v)])
            break

def basicstats(patients):
    print("# of unique patient: %d"%len(patients))
    target_icds = []
    for k, v in patients.iteritems():
        target_icds.extend([_[1] for _ in v[v.keys()[-1]]])
    print("# of unique target icds: %d"%len(list(set(target_icds))))
    print("Top occuring: ", Counter(target_icds).most_common(20))
    num_seq = sorted([len(enc.keys())-1 for p, enc in patients.iteritems()])
    print("FD:", np.histogram(num_seq, bins=[0, 1, 2, 5,10,20,50,100,500,1000,10000]))

def read_icd9_embeddings(embedding_path):
    token2idx = {'<PAD>': 0, '<UNK>': 1}
    _idx = 2
    f = open(embedding_path, 'r')
    embeddings = {}
    for line in f.readlines():
        line = line.replace('\n' ,'').strip().split(' ')
        if 'IDX_' in line[0]:
            token = line[0].replace('IDX_', '').replace('.', '')
            embeddings[token] = [float(_) for _ in line[1:]]
            token2idx[token] = _idx
            _idx += 1
        elif line[0] == '</s>':
            embeddings['</s>'] = [float(_) for _ in line[1:]]
    embeddings['<UNK>'] = embeddings['</s>']#np.zeros(300)
    embeddings['<PAD>'] = embeddings['</s>']#np.zeros(300)
    idx2token = {_[1]:_[0] for _ in token2idx.iteritems()}
    return (embeddings, token2idx, idx2token)


class ICD9DataBuilder():
    def __init__(self, data, label_map, token2idx):
        self.raw_data = data
        self.labels = label_map
        self.data = []
        self.token2idx = token2idx
        # Other preprocessing steps to come in this class
    
    def label_data(self):
        for patient, adms in self.raw_data.iteritems():
            last_encounter = adms.keys()[-1]
            for target_icd in adms[last_encounter]:
                if target_icd[1] in self.labels:
                    all_encounter_icds = []
                    for _hadm, _codes in adms.iteritems():
                        all_encounter_icds.append([_[1] for _ in _codes])
                    all_encounter_icds = all_encounter_icds[:-1]
                    if len(all_encounter_icds):
                        for i, enc in enumerate(all_encounter_icds):
                            all_encounter_icds[i] = [self.token2idx[_]  for _ in enc if (_ in self.token2idx.keys() and _!= '')]
                        self.data.append({'X': all_encounter_icds, 'Y': self.labels.index(target_icd[1])}) 

class ICD9Dataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, key):
        return (self.data_list[key]['X'], self.data_list[key]['Y'])


def padding_collation(batch):
    batch_list, label_list = [], []
    max_seq_len = np.max([len(datum[0]) for datum in batch])
    for datum in batch:
        padded_vec = [0 for i in range(max_seq_len-len(datum[0]))] + datum[0]
        batch_list.append(padded_vec)
        label_list.append(datum[1])
    return [batch_list, label_list]

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, batch_size, labels):
        super(LSTMModel, self).__init__()
        self.embed_dim = embed_dim
        #self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, len(labels))
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
    def init_hidden(self):
        return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)), Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))
    def forward(self, x, hidden):
        #x = self.embed(x)
        x = torch.transpose(x, 1, 0)
        x, _hidden  = self.lstm(x, hidden)
        x = x[-1, :, :].view(batch_size, -1)
        x = self.lin(x)
        return x

class Trainer():
    def __init__(self, train_loader, embeddings, token2idx, idx2token, test_loader, batch_size, labels):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.embeddings = embeddings
        self.token2idx = token2idx
        self.idx2token = idx2token
        self.batch_size = batch_size
        self.num_epochs = 100
        self.embedding_size = 300
        self.hidden_dim = 100
        self.lr = 1e-3
        self.model = LSTMModel(len(self.token2idx.keys()), self.embedding_size, self.hidden_dim, self.batch_size, labels)
        self.crit = nn.CrossEntropyLoss()
        self.opti = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.avg_pad = 0
        self.avg_tokens = 0
        self.avg_seq_len = 0

    def aggregate_embeddings(self, batch_x):
        batch_emb = np.ndarray((self.batch_size, len(batch_x[0]), self.embedding_size))
        #skipped = 0
        for i, enclist in enumerate(batch_x):
            #if np.sum([1 for _ in enclist if _ == 0]) > len(enclist)//2:
            #    skipped +=1
            #    continue
            for j, enc in enumerate(enclist):
                self.avg_tokens += 1
                if enc == 0:
                    batch_emb[i, j, :] = self.embeddings['<PAD>']
                    self.avg_pad +=1
                else:
                    token_embeddings = [self.embeddings[self.idx2token[icd]] for icd in enc]
                    if len(token_embeddings):
                        batch_emb[i, j, :] = np.mean(token_embeddings, axis=0) 
                    else:
                        batch_emb[i, j, :] = embeddings['<UNK>']
        return batch_emb

    def train(self):
        step = 0
        step_log = []
        loss_log = []
        val_acc_log = []
        val_loss_log = []

        for epoch in range(self.num_epochs):
            for batch in self.train_loader:
                if len(batch[0]) != self.batch_size:
                    continue

                self.model.zero_grad()
                batch_x, batch_y = batch[0], torch.LongTensor(batch[1])
                batch_emb = self.aggregate_embeddings(batch_x)
                x = Variable(torch.from_numpy(batch_emb).float())#.transpose(1,0)
                hidden = self.model.init_hidden()
                
                x = self.model(x, hidden)
                loss = self.crit(x, Variable(batch_y.view(-1)))
                #print ">>>>>>>>>>"
                #print x.size()
                #print x[0], batch_y.view(-1)[0]
                loss.backward()
                self.opti.step()
                # for param in self.model.parameters():
                #     print param[0]
                #     break
                if step % 100 == 0:
                    # print batch[0]
                    # print torch.max(x.data, 1)[1] 
                    # print batch_y
                    print "average paddings:", np.sum(self.avg_pad)/(np.sum(self.avg_tokens)*1.0)
                    self.avg_pad, self.avg_tokens = 0,0
                    self.model.eval()
                    val_acc, val_loss = self.evaluate()
                    self.model.train()
                    print("Step: %d, Loss: %.4f, Validation Acc: %.2f, Validation Loss: %.2f"%(step, loss.data[0], val_acc, val_loss))
                    step_log.append(step)
                    loss_log.append(loss.data[0])
                    val_acc_log.append(val_acc)
                    val_loss_log.append(val_loss)

                    self.lr *= 0.9
                    self.opti = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
                step += 1
        self.plot(step_log, loss_log, val_acc_log, val_loss_log)


    def evaluate(self):
        correct = 0
        total = 0
        avg_loss = 0
        for batch in self.test_loader:
            if len(batch[0]) != batch_size:
                continue
            batch_x, batch_y = batch[0], torch.LongTensor(batch[1])
            batch_emb = self.aggregate_embeddings(batch_x)
            x = Variable(torch.from_numpy(batch_emb).float())#.transpose(1,0)
            hidden = self.model.init_hidden()
            x = self.model(x, hidden)
            _, predicted = torch.max(x.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum()
            avg_loss = self.crit(x, Variable(batch_y.view(-1))).data[0]
                
        return (correct / float(total), np.mean(avg_loss))


    def plot(self, step_log, loss_log, val_acc_log, val_loss_log):
        plt.figure()
        plt.plot(step_log, loss_log)
        plt.title("Train Loss")
        plt.savefig("train_loss.png")
        plt.figure()
        plt.plot(step_log, val_acc_log)
        plt.title("Val Acs")
        plt.savefig("val_acc.png")
        plt.figure()
        plt.plot(step_log, val_loss_log)
        plt.title("Val Loss")
        plt.savefig("val_loss.png")



if __name__ == "__main__":
    base_path = '/media/disk3/disk3/'

    #patients, pat_adm_id = build_encounter_seq(os.path.join(base_path, 'mimic3/DIAGNOSES_ICD.csv'))
    #basicstats(patients)
    #exit()

    embeddings, token2idx, idx2token = read_icd9_embeddings(os.path.join(base_path, 'claims_codes_hs_300.txt'))

    labels = ['4019','41401','42731','4280','25000','2724','5849','V053','51881','V290','2720','53081','5990','2859','2449','V3000','2851','486','2762','496']

    datapath = os.path.join(base_path, 'mimic_labelled_sample.pickle')
    if os.path.exists(datapath):
        data = pickle.load(open(datapath, 'r'))['data']
    else:
        patients, pat_adm_id = build_encounter_seq(os.path.join(base_path, 'mimic3/DIAGNOSES_ICD.csv'))
        basicstats(patients)
        db = ICD9DataBuilder(patients, labels, token2idx)
        db.label_data()
        f = open(datapath, 'w')
        pickle.dump({'data': db.data}, f)
        f.close()
        data = db.data
    '''
    _bin = {1:0, 2:0}
    for i in [_['X'] for _ in data]:
        if len(i)>0:
            _bin[1] +=1
        elif len(i)>1:
            _bin[2] +=1
    print _bin
    exit()
    '''
    random.shuffle(data)

    _break = int(0.8*len(data))
    train_data = ICD9Dataset(data[:_break])
    test_data = ICD9Dataset(data[_break:])
    batch_size = 5
    train_loader = torch.utils.data.DataLoader(dataset=train_data, shuffle=True, 
                                batch_size=batch_size,collate_fn=padding_collation)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, shuffle=True, 
                                batch_size=batch_size,collate_fn=padding_collation)

    trainer = Trainer(train_loader, embeddings, token2idx, idx2token, test_loader, batch_size, labels) 
    trainer.train()
