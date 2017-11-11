import csv
from operations import itemgetter
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset

# will different hadm_id have same sequence_num continuation
#   find all subject id and confirm that
def build_encounter_seq(encounter_data_path):
    patients = {}
    pat_adm_id = {}
    _count = 0
    with open(encounter_data_path, 'rb') as csvf:
        csvreader = csv.reader(csvf, delimited=',', quotechar='"')
        for row in csvreader:
            _count += 1
            if _count > 10:
                break
            if patients.get(row[1]) is None:
                patients[row[1]] = {row[2]: [row[3], row[4]]}
                pat_adm_id[row[1]] = [row[2]]
            elif patients[row[1]].get(row[2]) is None::
                patients[row[1]][row[2]] = [row[3], row[4]]
                pat_adm_id[row[1]].append(row[2])
            else:
                patients[row[1]][row[2]].append([row[3], row[4]])
    for key in patients:
        patients[key] = {_[0]: _[1]  for _ in sorted(patients[key].iteritems())} 
    return (patients, pat_adm_id)

def POC_of_seq(patients, pat_adm_id):
    # POC of a subject with multiple hadm_id and check seq_num
    for key, val in pat_adm_id.iteritems():
        if len(val) > 1:
            for v in val:
                print(key, v, patients[(key, v)])
            break

def basicstats(patients):
    print "# of unique patient+adms: %d"%len(patients)
    target_icds = []
    for k, v in patients.iteritems():
        target_icds.extend([_[1] for _ in v[v.keys()[-1]]])
    print "# of unique target icds: %d"%list(set(target_icds))
    print "Top occuring: ", Counter(target_icds).most_common(20)
    num_seq = sorted([len(enc.keys())-1 for p, enc in patients.iteritems()])
    print "FD:", np.histogram(num_seq, bins=[0,10,20,50,100,500,1000,10000]) 

def read_icd9_embeddings(embedding_path):
    f = open(embedding_path, 'r')
    embeddings = {}
    for line in f.readlines():
        line = line.replace('\n' ,'').split(' ')
        if 'IDX_' in line[0]:
            embeddings[line[0].replace('IDX_', '').replace('.', '')] = line[1:]
    return embeddings

#base_path = '/media/disk3/disk3/'
#patients, pat_adm_id = build_encounter_seq(os.path.join(base_path, 'mimic3/DIAGNOSES_ICD.csv'))
#POC_of_seq(patients, pat_adm_id)
#basicstats(patients)
#TODO Preprocessing of windowing etc.
#Map ICD9 to embeddings (directly in pytorch)
#label samples
#split data and shuffle
#embeddings = read_icd9_embeddings(os.path.join(base_path, 'data/claims_codes_hs_300.txt'))


class ICD9DataBuilder():
    def __init__(self, data, label_map, embeddings, max_seq_len):
        self.raw_data = data
        self.labels = label_map
        self.raw_embeddings = embeddings
        self.embedding_size = len(self.raw_embeddings[0])
        self.max_seq_len = max_seq_len
        self.data = []

    def label_data(self):
        for patient, adms in self.data.iteritems():
            last_encounter = adms.keys()[-1]
            if target_icd in adms[last_encounter]:
                if target_icd[1] in label_map:
                    del adms.pop[last_encounter]
                    all_encounter_icds = []
                    for _hadm, _codes in adms.iteritems():
                        all_encounter_icds.append([_[1] for _ in _codes.iteritems()])
                    self.data.append({'X': self.aggregate_emebeddings(all_encounter_icds), 'Y': target_icd[1]}) 

    def aggregate_embeddings(self, icd_list):
        embed_tensor = torch.FloatTensor(icd_list, self.embedding_size)
        for idx, icds in enumerate(icd_list):
            embed_tensor[idx] = np.mean([self.raw_embeddings.get(icd, 'UNK') for icd in icds], axis=0)
        return embed_tensor

class ICD9Dataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, key):
        return (self.data_list[key]['X'], self.data_list[key]['Y'])

def padding_collation(batch):
    batch_list, label_list = [], []
    max_seq_len = np.max([len(datum[0]) for datum in batch])
    for datum in batch:
        padded_vec = np.pad(np.array, pad_width=((0, max_seq_len-len(datum[0])),
                        mode="constant", constant_values=0)
        batch_list.append(padded_vec)
        label_list.append(datum[1])
    return [torch.from_numpy(np.array(batch_list)), torch.LongTensor(label_list)]


