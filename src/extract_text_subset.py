import os
import pickle
import csv
from datetime import datetime
import numpy as np
from collections import Counter

base_path = '/misc/vlgscratch2/LecunGroup/laura/medical_notes'
from data_util import *
import re
import random
import time
import sklearn
import pprint

def select_only_discharge_notes(data):
    for d in data:
        if 'notes' in d:
            new_notes = []
            for note in d['notes']:
                if 'discharge' in note['note_type'].lower():
                    new_notes.append(note)
            d['notes'] = new_notes
    return data

def clean_and_extract_summary(s, inc_history=True):
    s = re.sub('\[\*\*.*\*\*\]|\\n|\s+', ' ', s).replace('  ', ' ').lower()
    s = extract_summary(s, inc_history)
    return s

def read_fork(data, op_queue):
    rawdata = []
    for key in data:
        if 'notes' in data[key]:
            data[key]['notes'] = sorted(data[key]['notes'], key=lambda x:datetime.strptime(x['date'], '%Y-%m-%d'))
            for n, note in enumerate(data[key]['notes']):
                data[key]['notes'][n]['note'] = clean_and_extract_summary(note['note'], True)

            rawdata.append(data[key])
    op_queue.put(rawdata)

def read_data_dump(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    for key in list(data.keys()):
        print(key)
        break
    get_data_stats(data)

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
        batch_data = {idx: data[idx] for idx in list(data.keys())[st_ind:end_ind]}
        print(len(batch_data.keys()))
        processes.append(multiprocessing.Process(target=read_fork, args=(
                                            batch_data,  output)))
    for p in processes:
        p.start()
    rawdata = []
    results = [output.get() for p in processes]
    for result in results:
        rawdata.extend(result)

    return rawdata


if __name__ == "__main__":
    print("Reading data")
    rawdata = read_data_dump(os.path.join(base_path, 'notes_dump.pkl'))
    rawdata = select_only_discharge_notes(rawdata)
    get_data_stats_2(rawdata)
    f = open(os.path.join(base_path, 'notes_dump_diagnosis_only_note_extract_inc_history.pkl'), 'wb')
    pickle.dump({'data':rawdata}, f)
    f.close()
