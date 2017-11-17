import os
import pickle
import csv
from datetime import datetime
import numpy as np
from collections import Counter

base_path = '/media/disk3/disk3'
from sklearn.feature_extraction import stop_words
import re
import random

import time
import sklearn

global_time = None
def gettime():
    global global_time
    if global_time is None:
        global_time = time.time()
        diff = 0
    else:
        diff = (time.time()-global_time)/60.0
        global_time = time.time()
    return diff

from nltk.tokenize import word_tokenize
def clean_str_no_stopwords(s, stopwords=stop_words.ENGLISH_STOP_WORDS):
    s = re.sub('\[\*\*.*\*\*\]|\\n|\s+|[^\w\s]', ' ', s).replace('  ', ' ').lower()#.split() 
    if stopwords is not None:
        s = [w if w not in stopwords else 'unknown' for w in word_tokenize(s)]
    else:
        s = [w for w in word_tokenize(s)]
    return s

def read_fork(data, op_queue):
    rawdata = []
    for key in data:
        if 'notes' in data[key]:
            x = sorted(data[key]['notes'], key=lambda x:datetime.strptime(x['date'], '%Y-%m-%d'))
            x = [clean_str_no_stopwords(note['note']) for note in data[key]['notes']]
            x = [note for note in x if note != []]
            data[key]['notes'] = x 
            rawdata.append(data[key])
    op_queue.put(rawdata)

def read_data_dump(data_path):
    with open(data_path, 'r') as f:
        data = pickle.load(f)
    data = {k:data[k] for k in data.keys()[:5000]} 

    import multiprocessing
    num_workers = 10
    output = multiprocessing.Queue()
    batch_size = len(data.keys()) // num_workers
    processes = []
    for _ in range(num_workers):
        st_ind = _*batch_size
        end_ind = st_ind+batch_size
        if _ == num_workers-1 and end_ind < len(data.keys()):
            end_ind = len(data.keys())
        batch_data = {idx: data[idx] for idx in data.keys()[st_ind:end_ind]}
        print len(batch_data.keys())
        processes.append(multiprocessing.Process(target=read_fork, args=(
                                            batch_data,  output)))
    for p in processes:
        p.start()
    rawdata = []
    results = [output.get() for p in processes]
    for result in results:
        rawdata.extend(result)

    return rawdata

def read_embeddings(vecidx_path, vec_path):
    words = []
    with open(vecidx_path, 'r') as f:
        tsvreader = csv.reader(f, delimiter='\t')
        for i,row in enumerate(tsvreader):
            words.append(row[0])

    vecs = np.ndarray((len(words), 400))
    with open(vec_path, 'r') as f:
        tsvreader = csv.reader(f, delimiter='\t')
        for i,row in enumerate(tsvreader):
            vecs[i,:] = row[:400]
    vecs = sklearn.preprocessing.normalize(vecs)
    word2vec = {_:vecs[i,:].reshape(-1) for i,_ in enumerate(words)}
    return word2vec

def filter_labels(data):
    labels = [key['labels']['icd'][0] for key in data]
    labels = [_[0] for _ in list(Counter(labels).most_common(20))]
    return labels
    
def filter_data_by_labels(data, labels): 
    return [key for key in data if key['labels']['icd'][0] in labels]

def get_vocab(data):
    vocab = []
    for key in data:
        for note in key['notes']:
            vocab.extend(note)
    print vocab[:200]
    vocab = [_[0] for _ in list(Counter(vocab).most_common(2000))]
    return vocab

def filter_data_by_vocab(data, vocab):
    for _,key in enumerate(data):
        for i, note in enumerate(key['notes']):
            data[_]['notes'][i] = " ".join([word if word in vocab else 'unknown' for word in key['notes'][i]])
    return data

def filter_embeddings(vocab):
    pretrained = read_embeddings(os.path.join(base_path, 'ri-3gram-400-tsv/vocab.tsv'), 
                                         os.path.join(base_path, 'ri-3gram-400-tsv/vectors.tsv'))
    csvf = open(os.path.join(base_path, 'ri-3gram-400-tsv/filtered_embeddings.tsv'), 'w')
    writer = csv.writer(csvf, delimiter='\t')
    pretrained_filtered = {}
    writer.writerow(['pad']+list(pretrained['pad']))
    writer.writerow(['unknown']+list(pretrained['unknown']))
    for _,word in enumerate(vocab):
        vec = pretrained.get(word, None)
        if vec is not None:
            writer.writerow([word]+list(pretrained[word]))
    csvf.close()

if __name__ == "__main__":
    #reading -> sort by date, remove de-id, puncts, tokenize , stopwords
    '''    
    print "Reading data"
    rawdata = read_data_dump(os.path.join(base_path, 'notes_dump.pkl'))
    print "data size:", len(rawdata)
    print "Filtering labels"
    labels = filter_labels(rawdata)
    print "num labels:", len(labels)
    print "filter_data_by_labels"
    rawdata = filter_data_by_labels(rawdata, labels)
    print "data size:", len(rawdata)
    print "vocab..."
    vocab = get_vocab(rawdata)
    print "Vocab size:", len(vocab)
    f = open(os.path.join(base_path, 'filtered_vocab_10000.txt'), 'w')
    for v in vocab:
        print >>f, v
    f.close()
    print "filter_data_by_vocab"
    rawdata = filter_data_by_vocab(rawdata, vocab)
    f = open(os.path.join(base_path, 'notes_dump_cleaned.pkl'), 'w')
    pickle.dump({'data':rawdata}, f)
    f.close()
    print "filter_embeddings"
    '''    
    
    f = open(os.path.join(base_path, 'filtered_vocab_10000.txt'), 'r')
    vocab = []
    for line in f.readlines():
        vocab.append(line.replace('\n', ''))
 
    filter_embeddings(vocab)

