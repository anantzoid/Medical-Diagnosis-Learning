import os
import csv
import numpy as np
from collections import Counter
import argparse
from preprocess_helpers import *

base_path = '/Users/lauragraesser/Documents/NYU_Courses/medical_data'

parser = argparse.ArgumentParser(description='MIMIC III notes data preparation')
parser.add_argument('--data', type=str, default='/Users/lauragraesser/Documents/NYU_Courses/medical_data', help="folder where data is located")
parser.add_argument('--numlabels', type=int, default=10,
                    help='Number of distinct ICD9 codes to use')
parser.add_argument('--ICDcodelength', type=int, default=4,
                    help='How many of the digits of the ICD code to use as a label')
parser.add_argument('--notestypes', type=str, default='discharge summary',
                    help='Types of notes to include')
parser.add_argument('--notescontent', type=int, default=2,
                    help='What part of the note to include')
parser.add_argument('--preprocessing', type=str, default='replace numbers',
                    help='What preprocessing to do on the text')
parser.add_argument('--vocabcountthreshold', type=int, default=10,
                    help='Only include words with count > threshold in vocabulary')
parser.add_argument('--mapunk', type=bool, default=False,
                    help='Whether to map OOV words to words in vocab using edit dist')
args = parser.parse_args()
print(args)

# Processing params
base_path = args.data
num_labels = args.numlabels # Number of distinct ICD9 codes to use
ICD_code_length = args.ICDcodelength # How many of the digits of the ICD code to use as a label
note_types = args.notestypes.split(',') # Types of notes to include
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

# Load ICD data and select
diagnosis = get_diagnosis(os.path.join(base_path, 'DIAGNOSES_ICD.csv'), ICD_code_length)
print("Total number of HADM_ID: {}".format(len(diagnosis)))
print(diagnosis['172335'])
top_diagnoses = get_top_diagnoses(diagnosis, num_labels)
print("Top diagnoses: {}".format(top_diagnoses))
processed_diagnoses = remove_diagnoses_not_intopK(diagnosis, top_diagnoses)
print(processed_diagnoses['172335'])
