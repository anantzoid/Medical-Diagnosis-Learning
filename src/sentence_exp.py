exp_name = 'summaries_50label_vocab50'

import torch
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(0)
multi_gpu = False#True

import sys
import pickle
import random
from collections import Counter
sys.path.append('/home/ag4508/Medical-Diagnosis-Learning/src')
from data_util import *
from models import *
#from util_icu_train import get_labels

import csv    
#get_top_labels = lambda path: [row[0] for row in  csv.reader(open(path, "r"), delimiter=",")]
#load_summaries = lambda path: [{'text':row[1], 'label':row[2].split(',') for row in  csv.reader(open(path, "r"), delimiter=",", quotechar='"')}]
def get_top_labels(path):
    labels = [row[0] for row in  csv.reader(open(path, "r"), delimiter=",")]
    return {i:_ for _,i in enumerate(labels)}

def load_summaries(path):
    data = []
    for row in  csv.reader(open(path, "r"), delimiter=",", quotechar='"'):
        if row[1] == '':
            continue
        data.append({
            'text': row[1],
            'label': row[2].split(',')
        })
    return data

#label_map = get_labels('/misc/vlgscratch2/LecunGroup/anant/nlp/labels.txt')
#training_set = load_data_csv('/misc/vlgscratch2/LecunGroup/anant/nlp/notes_sample_5class.csv', label_map)

label_path = '../data/top50_labels.csv'
label_map = get_top_labels(label_path) 

data_path = '../data/summaries_labels.csv'
training_set = load_summaries(data_path)

analytics = False
if analytics:
    text = [_[0] for _ in training_set]
    lentext = [len(_[0]) for _ in training_set]
    import numpy as np
    print np.histogram(lentext, bins=[0,1,5,10,50,100,500,1000,5000])
    print text[np.argmax(lentext)]
    print np.mean(lentext)
    print np.max(lentext)
    print np.min(lentext)
    print("FD of labels:", Counter([i for _ in training_set for i in _[1]]))
    exit()

#training_set = training_set[:1000]
PADDING = "<PAD>"
UNKNOWN = "<UNK>"
max_seq_length = 500
min_vocab_threshold = 50
batch_size = 64
num_workers = 4
embed_dim = 100
hidden_dim = 300
lr = 1e-2
num_epochs = 100

word_to_ix, vocab_size, word_counter = build_dictionary([training_set], PADDING, UNKNOWN, vocab_threshold=min_vocab_threshold)
sentences_to_padded_index_sequences(word_to_ix, training_set, max_seq_length, PADDING, UNKNOWN, label_map)
print "Vocab size: %d"%vocab_size

random.shuffle(training_set)
val_set = training_set[int(0.8*len(training_set)):]
training_set = training_set[:int(0.8*len(training_set))]

train_loader = torch.utils.data.DataLoader(dataset= TextData(training_set), batch_size=batch_size, shuffle=True, 
                                                           num_workers=num_workers, collate_fn=sent_batch_collate)
val_loader = torch.utils.data.DataLoader(dataset= TextData(val_set), batch_size=batch_size, shuffle=True, 
                                                           num_workers=num_workers, collate_fn=sent_batch_collate)


model = LSTMModel(vocab_size, embed_dim, hidden_dim, label_map, batch_size, use_cuda)
opti = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
#crit = nn.CrossEntropyLoss()
crit = nn.BCEWithLogitsLoss()

if use_cuda:
    model.cuda()
    crit.cuda()
    if multi_gpu:
        model = nn.DataParallel(model, device_ids=(0, 1)) 

from eval import *    

step = 0
step_log = []
loss_log = []
val_acc_log = []
val_loss_log = []
train_acc_log = []

for nu_ep in range(num_epochs):
    for batch in train_loader:
        if batch[0].size(0) != batch_size:
            continue
        model.zero_grad()
        x = Variable(batch[0])
        y = Variable(batch[1])#.view(-1))
        if use_cuda:
            x, y = x.cuda(), y.cuda()

        #hidden = model.init_hidden()
        x = model(x)#, hidden)
        loss = crit(x, y)
        loss.backward()
        opti.step()

        if step % 10 == 0:
            model.eval()
            #_, predicted = torch.max(x.data>0.5, 1)    
            predicted = x.data > 0.5
            train_acc = (predicted.float() == y.data).sum() / float(batch[1].size(0))

            val_acc, val_loss, pred_vals = evaluate(model, val_loader, batch_size, crit, use_cuda)
            model.train()        
            print("Step: %d, Epoch: %d, Loss: %.4f, Train Acc: %.2f, Validation Acc: %.2f, Val loss: %.2f"%(step, nu_ep, loss.data[0], train_acc, val_acc, val_loss))
            step_log.append(step)
            loss_log.append(loss.data[0])
            train_acc_log.append(train_acc)
            val_acc_log.append(val_acc)
            val_loss_log.append(val_loss)
        #if step % 100 == 0:
        #    print(pred_vals) 
        step += 1
    
    if nu_ep%10==0:
    #if step % 10000 == 0:
        lr *= 0.9
        opti = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
        torch.save(model.state_dict(), '/misc/vlgscratch2/LecunGroup/anant/nlp/model_%s.pth'%exp_name)


    f = open('/misc/vlgscratch2/LecunGroup/anant/nlp/results_%s.pkl'%exp_name, 'w')
    pickle.dump({'step_log': step_log, 'loss_log': loss_log, 'val_loss_log': val_loss_log, 'val_loss_log': val_loss_log, 'train_acc_log': train_acc_log, 'val_acc_log': val_acc_log}, f)
    f.close()
#import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')

#plt.plot(step_log, loss_log)



