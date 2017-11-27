import os
import csv
import numpy as np
from collections import Counter
import argparse
import pprint
import pickle
import time
from preprocess_helpers import *
from data_util import get_data_stats

parser = argparse.ArgumentParser(description='MIMIC III notes data preparation')
parser.add_argument('--data', type=str, default='/Users/lauragraesser/Documents/NYU_Courses/medical_data', help="folder where data is located")
parser.add_argument('--notesfile', type=str, default='NOTEEVENTS.csv', help="notes data file")
parser.add_argument('--procdatafile', type=str, default='base', help="processed data file name")
parser.add_argument('--numlabels', type=int, default=10,
                    help='Number of distinct ICD9 codes to use')
parser.add_argument('--ICDcodelength', type=int, default=5,
                    help='How many of the digits of the ICD code to use as a label')
parser.add_argument('--notestypes', type=str, default='discharge summary',
                    help='Types of notes to include')
parser.add_argument('--notescontent', type=int, default=3,
                    help='What part of the note to include')
parser.add_argument('--preprocessing', type=str, default='add space,remove brackets,replace numbers,replace break',
                    help='What preprocessing to do on the text')
parser.add_argument('--vocabcountthreshold', type=int, default=5,
                    help='Only include words with count > threshold in vocabulary')
parser.add_argument('--mapunk', type=int, default=0,
                    help='Whether to map OOV words to words in vocab using edit dist and how')
parser.add_argument('--generatesplits', type=int, default=0,
                    help='Whether to generate a new split of the data for train-valid-test')
args = parser.parse_args()
print(args)
print()

# Processing params
base_path = args.data
notes_file = args.notesfile
num_labels = args.numlabels # Number of distinct ICD9 codes to use
ICD_code_length = args.ICDcodelength # How many of the digits of the ICD code to use as a label
note_types = args.notestypes.split(',') # Types of notes to include
print("Note types included: {}".format(note_types))
notes_content = args.notescontent
'''
Notes content options
    1: whole note
    2: short discharge diagnosis and final diagnosis
    3: all discharge diagnosis and final diagnosis
    4: all discharge diagnosis and final diagnosis and present history of illness
'''
preprocessing = args.preprocessing.split(',')
vocab_count_threshold = args.vocabcountthreshold # Only include words with count > threshold in vocabulary
map_unk_using_edit_dist = args.mapunk # Whether to map OOV words to words in vocab using edit dist

# Load ICD data and process ICD codes
total_start = time.time()
diagnosis = get_diagnosis(os.path.join(base_path, 'DIAGNOSES_ICD.csv'), ICD_code_length)
print("Total number of HADM_ID: {}\n".format(len(diagnosis)))
print("Example: raw data")
print(diagnosis['178393'])
print()
if args.generatesplits:
    print("Generating HADM ID splits...")
    splits = split_hadm_ids(diagnosis)
    with open(os.path.join(base_path, "hadm_id_train_valid_test_splits.pkl"), 'wb') as f:
        pickle.dump(splits, f)
splits = pickle.load(open(os.path.join(base_path, "hadm_id_train_valid_test_splits.pkl"), 'rb'))
print("Length train: {} valid: {} test: {}".format(len(splits[0]),len(splits[1]),len(splits[2])))
top_diagnoses = get_top_diagnoses(diagnosis, num_labels)
print("Top diagnoses: {}\n".format(top_diagnoses))
processed_diagnoses = remove_diagnoses_not_intopK(diagnosis, top_diagnoses)
print("Example after processing ICD codes")
print(processed_diagnoses['178393'])
print()
processed_diagnoses = remove_blank_examples(processed_diagnoses)
print("Total number of HADM_ID after ICD processing: {}\n".format(len(processed_diagnoses)))
try:
    print("Example after removing blanks")
    print(processed_diagnoses['178393'])
except:
    print("No key after processing: 178393")
print()

# Load notes data and process
data = build_notes(os.path.join(base_path, notes_file), processed_diagnoses, note_types)
print("Data stats after loading data")
get_data_stats(data)
print()
data = process_text(data, notes_content, preprocessing)
get_data_stats(data)
print()
data = convert_format(data)
print("FINAL NUMBER OF DATAPOINTS: {}".format(len(data)))
print("EXAMPLES")
print(data[0])
print()
print(data[1])
print()
# split datasets
train_data, valid_data, test_data = split_data(data, splits)
print("Length of split data: train: {}, valid: {}, test: {}".format(
    len(train_data), len(valid_data), len(test_data)
))
# Tokenize (using python tokenization as spacy is slow)
start = time.time()
train_data = tokenize_by_sent_alt(train_data)
end = time.time()
print("Time for python tokenization on training data: {:.3f}".format(end - start))
valid_data = tokenize_by_sent_alt(valid_data)
test_data = tokenize_by_sent_alt(test_data)
print("EXAMPLES AFTER TOKENIZATION")
print(len(train_data[0][1]))
print(train_data[0])
print()
print(len(valid_data[0][1]))
print(valid_data[0])
print()
# Vocabify
vocab = extract_vocab(train_data, vocab_count_threshold)
print("Vocabifying training data...")
train_data = vocabify_text(train_data, vocab, map_unk_using_edit_dist)
print("Vocabifying valid data...")
valid_data = vocabify_text(valid_data, vocab, map_unk_using_edit_dist)
print("Vocabifying test data...")
test_data = vocabify_text(test_data, vocab, map_unk_using_edit_dist)
print("EXAMPLES AFTER VOCAB EXTRACTION AND DATA PREP")
print(len(train_data[0][1]))
print(train_data[0])
print()
print(len(valid_data[0][1]))
print(valid_data[0])
print()
# # write to files
with open(os.path.join(base_path, args.procdatafile + '_train_data.pkl'), 'wb') as f:
    pickle.dump(train_data, f)
with open(os.path.join(base_path, args.procdatafile + '_valid_data.pkl'), 'wb') as f:
    pickle.dump(valid_data, f)
with open(os.path.join(base_path, args.procdatafile + '_test_data.pkl'), 'wb') as f:
    pickle.dump(test_data, f)
print("Written data to file")
total_time = time.time() - total_start
print("Total time to process data {:.2f}".format(total_time))
