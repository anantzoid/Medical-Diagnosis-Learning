import csv
import unicodedata
import re
import random
from collections import Counter

import collections
import numpy as np
import torch
import re

def count_length(datapoint):
    length = 0
    if 'notes' in datapoint:
        for note in datapoint['notes']:
            if isinstance(note['note'], str):
                length += len(note['note'])
            else:
                length += sum([len(word) + 1 for word in note['note']])
    return length

def calc_average_length(data):
    lengths = []
    if isinstance(data, list):
        for d in data:
            lengths.append(count_length(d))
    else:
        for key in data:
            lengths.append(count_length(data[key]))
    return np.mean(lengths)

def get_data_stats(data):
    print ("No. of data points:", len(data.keys()))
    print ("Data points with no notes:", np.sum([1 for _ in data.keys() if data[_].get('notes') is None]))
    print ("Average No. of notes per Hadm_id:", np.mean([len(data[_].get('notes', [])) for _ in data.keys()]))
    num_notes = 0
    num_discharge_notes = 0
    nt = []
    for _ in data.keys():
        if data[_].get('notes'):
            for note in data[_]['notes']:
                nt.append(note["note_type"])
                num_notes += 1
                if 'discharge' in note['note_type'].lower():
                    num_discharge_notes += 1
    print("{} notes, {} discharge notes, ratio: {}".format(
        num_notes, num_discharge_notes, num_discharge_notes / (num_notes * 1.) * 100))
    print("Average length of note (chars): {:.2f}".format(calc_average_length(data)))
    print(dict(Counter(nt).most_common(30)))
    return None

def get_data_stats_2(data):
    print ("No. of data points:", len(data))
    print ("Data points with no notes:", np.sum([1 for _ in data if _.get('notes') is None]))
    print ("Average No. of notes per Hadm_id:", np.mean([len(_.get('notes', [])) for _ in data]))
    num_notes = 0
    num_discharge_notes = 0
    nt = []
    for _ in data:
        if _.get('notes'):
            for note in _['notes']:
                nt.append(note["note_type"])
                num_notes += 1
                if 'discharge' in note['note_type'].lower():
                    num_discharge_notes += 1
    print("{} notes, {} discharge notes, ratio: {}".format(
        num_notes, num_discharge_notes, num_discharge_notes / (num_notes * 1.) * 100))
    print("Average length of note (chars): {:.2f}".format(calc_average_length(data)))
    print(dict(Counter(nt).most_common(30)))
    return None

def extract_summary(text, inc_hist=True):
    newtext = ''
    if inc_hist:
        if 'history of present illness' in text and 'past medical history' in text:
            # print("Found present and past illness")
            newtext+= text[text.index('history of present illness'):text.index('past medical history')]
        elif 'history of present illness' in text:
            # print("Found present")
            start_idx = text.index('history of present illness')
            end_idx = min(text.index('history of present illness') + 1500, len(text))
            newtext += text[start_idx:end_idx]
    if 'final diagnosis' in text or 'discharge diagnosis' in text:
        if 'final diagnosis' in text:
            newtext += text[text.index('final diagnosis'):]
        if 'discharge diagnosis' in text:
            newtext += text[text.index('discharge diagnosis'):]
    return newtext

def load_data_csv(path, label_map):
    data= []
    line = 0
    with open(path, "r") as csvf:
        csvreader = csv.reader(csvf, delimiter=",")
        _data = {}
        for row in csvreader:
            line +=1
            #_text = row[0].replace("\n", "")
            _text = row[0]
            _text = extract_summary(_text.lower())
            _text = re.sub('\[\*\*.*\*\*\]|\s+|_', ' ', _text).replace('  ', ' ')#.lower()#.split()
            _text = _text.split(".\n")
            for _t in _text:
                if len(_t) > 5:
                    data.append({'text':_t, 'label':label_map[row[1]]})
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

def sent_batch_collate(batch):
    data_list = []
    label_list = []

    max_length = np.max([len(_[0]) for _ in batch])
    # padding
    for datum in batch:
        padded_str = datum[0] + [0]*(max_length-len(datum[0]))
        data_list.append(padded_str)
        label_list.append(datum[1])
    return (torch.from_numpy(np.array(data_list)).long(), torch.from_numpy(np.array(label_list)).float())


def sentences_to_padded_index_sequences(word_indices, dataset, max_seq_length, PADDING, UNKNOWN, label_map):
    """
    Annotate datasets with feature vectors. Adding right-sided padding.
    """
    for example in dataset:
        token_sequence = tokenize(example['text'])
        example['text_index_sequence'] = [0]*(min(len(token_sequence), max_seq_length))
        for i in range(len(example['text_index_sequence'])):
            if token_sequence[i] in word_indices:
                index = word_indices[token_sequence[i]]
            else:
                index = word_indices[UNKNOWN]
            example['text_index_sequence'][i] = index
        label_onehot = np.zeros((len(label_map.keys())))
        for la in example['label']:
            label_onehot[label_map[la]] = 1
        example['label'] = label_onehot
        '''
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
        '''
