import os
import csv
import numpy as np
from collections import Counter
import argparse
import pprint
from preprocess_helpers import *
from data_util import get_data_stats

parser = argparse.ArgumentParser(description='MIMIC III notes data preparation')
parser.add_argument('--data', type=str, default='/Users/lauragraesser/Documents/NYU_Courses/medical_data', help="folder where data is located")
parser.add_argument('--notesfile', type=str, default='NOTEEVENTS.csv', help="notes data file")
parser.add_argument('--numlabels', type=int, default=10,
                    help='Number of distinct ICD9 codes to use')
parser.add_argument('--ICDcodelength', type=int, default=4,
                    help='How many of the digits of the ICD code to use as a label')
parser.add_argument('--notestypes', type=str, default='discharge summary',
                    help='Types of notes to include')
parser.add_argument('--notescontent', type=int, default=2,
                    help='What part of the note to include')
parser.add_argument('--preprocessing', type=str, default='replace numbers,replace break',
                    help='What preprocessing to do on the text')
parser.add_argument('--vocabcountthreshold', type=int, default=10,
                    help='Only include words with count > threshold in vocabulary')
parser.add_argument('--mapunk', type=bool, default=False,
                    help='Whether to map OOV words to words in vocab using edit dist')
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
diagnosis = get_diagnosis(os.path.join(base_path, 'DIAGNOSES_ICD.csv'), ICD_code_length)
print("Total number of HADM_ID: {}\n".format(len(diagnosis)))
print("Example: raw data")
print(diagnosis['172335'])
print()
top_diagnoses = get_top_diagnoses(diagnosis, num_labels)
print("Top diagnoses: {}\n".format(top_diagnoses))
processed_diagnoses = remove_diagnoses_not_intopK(diagnosis, top_diagnoses)
print("Example after processing ICD codes")
print(processed_diagnoses['172335'])
print()
processed_diagnoses = remove_blank_examples(processed_diagnoses)
print("Total number of HADM_ID after ICD processing: {}\n".format(len(processed_diagnoses)))
try:
    print("Example after removing blanks")
    print(processed_diagnoses['172335'])
except:
    print("No key after processing: 172335")
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
print(data[2])
print()
write_to_file(os.path.join(base_path, 'processed_data.csv'), data)
