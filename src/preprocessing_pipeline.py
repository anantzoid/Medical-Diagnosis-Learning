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
def clean_str_no_stopwords(s, embed, stopwords=stop_words.ENGLISH_STOP_WORDS):
    s = re.sub('\[\*\*.*\*\*\]|\\n|\s+|[^\w\s]', ' ', s).replace('  ', ' ').lower()#.split() 
    if stopwords is not None:
        s = [w if w in stopwords else 'unknown' for w in word_tokenize(s)]
    else:
        s = [w for w in word_tokenize(s)]
    return s

def read_data_dump(data_path):
    with open(data_path, 'r') as f:
        data = pickle.load(f)
    rawdata = []
    for key in data:
        if 'notes' in data[key]:
            x = sorted(data[key]['notes'], key=lambda x:datetime.strptime(x['date'], '%Y-%m-%d'))
            x = [clean_str_no_stopwords(note['note'], embed) for note in data[key]['notes']]
            x = [note for note in x if note != []]
            data[key]['notes'] = x 
            rawdata.append(data[key])
    return data

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
    labels = [data[key]['labels']['icd'][0] for key in data]
    labels = list(Counter(labels).most_common(20))
    return labels
    
filter_data_by_labels = lambda(data, labels): [data[key] for key in data if data[key]['labels']['icd'][0] in labels]

def get_vocab(data):
    vocab = []
    for key in data.keys():
        for note in data[key]['notes']:
            vocab.extend(note['note'])
    vocab = list(Counter(vocab).most_common(10000))
    return vocab

def filter_data_by_vocab(data, vocab):
    for key in data.keys():
        for i, note in enumerate(data[key]['notes']):
            data[key]['notes'][i] = " ".join([word if word in vocab else 'unknown' for word in data[key]['notes'][i]])
    return data

def filter_embeddings(vocab):
    pretrained = read_embeddings(os.path.join(base_path, 'ri-3gram-400-tsv/vocab.tsv'), 
                                         os.path.join(base_path, 'ri-3gram-400-tsv/vectors.tsv'))
    csvf = open(os.path.join(base_path, 'ri-3gram-400-tsv/filtered_embeddings.tsv'), 'w')
    writer = csv.writer(csvf, delimiter='\t')
    pretrained_filtered = {}
    pretrained_filtered['unknown'] = pretrained['unknown']
    pretrained_filtered['pad'] = pretrained['pad']
    for _,word in enumerate(vocabs):
        vec = pretrained.get(word, None)
        if vec is not None:
            writer.writerow([word]+list(pretrained[word]))
    csvf.close()

if __name__ == "__main__":
    #reading -> sort by date, remove de-id, puncts, tokenize , stopwords
    print "Reading data"
    rawdata = read_data_dump(os.path.join(base_path, 'notes_dump.pkl'))
    rawdata = rawdata[:100]
    print "Filtering labels"
    labels = filter_labels(rawdata)
    print "filter_data_by_labels"
    rawdata = filter_data_by_labels(rawdata, labels)
    print "vocab..."
    vocab = get_vocab(data)
    f = open(os.path.join(base_path, 'filtered_vocab_10000.txt'), 'w')
    for v in vocab:
        print >>f, v
    f.close()
    print "filter_data_by_vocab"
    data = filter_data_by_vocab(data, vocab)
    f = open(os.path.join(base_path, 'notes_dump_cleaned.pkl'), 'w')
    pickle.dump({'data':data}, f)
    f.close()
    print "filter_embeddings"
    filter_embeddings(vocab)

