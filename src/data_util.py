import csv
import unicodedata
import re
import random
from collections import Counter

import collections
import numpy as np
import torch
import re

def extract_summary(text):
    newtext = ''
    if 'final diagnosis' in text or 'discharge diagnosis' in text:
        if 'final diagnosis' in text:
            newtext += text[text.index('final diagnosis'):]
        if 'discharge diagnosis' in text:
            newtext += text[text.index('discharge diagnosis'):]
    return newtext
             

def load_data_csv(path, easy_label_map):
    data= []
    line = 0
    with open(path, "r") as csvf:
        csvreader = csv.reader(csvf, delimiter=",")
        _data = {}        
        for row in csvreader:
            line +=1
            #_text = row[0].replace("\n", "")
            _text = extract_summary(_text) 
            _text = re.sub('\[\*\*.*\*\*\]|\s+|_', ' ', row[0]).replace('  ', ' ').lower()#.split() 
            _text = _text.split(".\n")
            for _t in _text:
                if len(_t) > 5:
                    data.append({'text':_t, 'label':easy_label_map[row[1]]})

    random.shuffle(data)
    print("# of data samples:%d"%len(data))
    return data

def tokenize(string):
    return string.split()

def build_dictionary(training_datasets, PADDING, UNKNOWN, vocab_threshold):
    """
    Extract vocabulary and build dictionary.
    """  
    word_counter = collections.Counter()
    for i, dataset in enumerate(training_datasets):
        for example in dataset:
            word_counter.update(tokenize(example['text']))
    
    vocabulary = set([word for word in word_counter if word_counter[word] > vocab_threshold])
    vocabulary = list(vocabulary)
    vocabulary = [PADDING, UNKNOWN] + vocabulary
        
    word_indices = dict(zip(vocabulary, range(len(vocabulary))))

    return word_indices, len(vocabulary), word_counter

def sentences_to_padded_index_sequences(word_indices, datasets, max_seq_length, PADDING, UNKNOWN):
    """
    Annotate datasets with feature vectors. Adding right-sided padding. 
    """
    for i, dataset in enumerate(datasets):
        for example in dataset:
            example['text_index_sequence'] = torch.zeros(max_seq_length)

            token_sequence = tokenize(example['text'])
            padding = max_seq_length - len(token_sequence)

            for i in range(max_seq_length):
                if i >= len(token_sequence):
                    index = word_indices[PADDING]
                    pass
                else:
                    if token_sequence[i] in word_indices:
                        index = word_indices[token_sequence[i]]
                    else:
                        index = word_indices[UNKNOWN]
                example['text_index_sequence'][i] = index

            example['text_index_sequence'] = example['text_index_sequence'].long().view(1,-1)
            if type(example['label']) == int:
                example['label'] = torch.LongTensor([example['label']])


