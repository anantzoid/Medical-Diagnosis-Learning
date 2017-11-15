'''
    Script to aggregate notes over hadm_id & other helpers
'''

import csv
import os
from collections import Counter
import json
import pickle
import numpy as np
def build_notes_dump(notes_path, data):
    with open(notes_path, 'rb') as csvf:
        csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
        skip_header = next(csvreader)
        for row in csvreader:
            if data.get(row[2]):
                # Skipping is ISERROR
                if isinstance(row[-2], int) and int(row[-2]) == 1:
                    continue
                note_dict = {"note_type": row[6],
                            "description": row[7],
                            "note": row[-1],
                            "date": row[3]
                            }
                if data[row[2]].get('notes') is None:
                    data[row[2]]['notes'] = [note_dict]
                else:
                    data[row[2]]['notes'].append(note_dict)
    return data

def get_diagnosis(label_path):
    with open(label_path, 'rb') as csvf:
        csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
        skip_header = next(csvreader)
        icd = {}
        #_c = 0
        for row in csvreader:
            #_c += 1
            #if _c > 50:
            #    break
            if icd.get(row[2]) is None:
                icd[row[2]] = {'labels':{'icd': [row[-1]], 'seq_no': [row[-2]]}, 'pat_id': row[1]}
            else:
                icd[row[2]]['labels']['icd'].append(row[-1])
                icd[row[2]]['labels']['seq_no'].append(row[-2])
    return icd 

def make_data_dump():
    base_path = '/media/disk3/disk3/mimic3'
    diagnosis = get_diagnosis(os.path.join(base_path, 'DIAGNOSES_ICD.csv'))
    diagnosis = build_notes_dump(os.path.join(base_path, 'NOTEEVENTS.csv'), diagnosis)

    with open('/media/disk3/disk3/notes_dump.pkl', 'w') as f:
        pickle.dump(diagnosis, f)
    f.close()

'''
    avg. #of notes per hadmid
    avg. #of labels per hadmid
    freq of note types for all
    avg. note size (# of words)
    total vocab size (excluding [*.*])
'''
def get_data_stats():
    with open('/media/disk3/disk3/notes_dump.pkl') as f:
        data = pickle.load(f)
    print ("No. of data points:", len(data.keys()))
    print ("Data points with no notes:", np.sum([1 for _ in data.keys() if data[_].get('notes') is None]))
    print ("Average No. of notes per Hadm_id:", np.mean([len(data[_].get('notes', [])) for _ in data.keys()]))
    nt = []
    for _ in data.keys():
        if data[_].get('notes'):
            for note in data[_]['notes']:
                nt.append(note["note_type"])
    print dict(Counter(nt).most_common(30))
    return None

if __name__ == "__main__":
    get_data_stats()
