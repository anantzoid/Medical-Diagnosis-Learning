import re
import csv
from collections import Counter

def get_diagnosis(label_path, icd_length):
    with open(label_path, 'r') as csvf:
        csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
        skip_header = next(csvreader)
        icd = {}
        for row in csvreader:
            if icd.get(row[2]) is None:
                icd[row[2]] = {'labels':{'icd': [row[-1][:icd_length]], 'seq_no': [row[-2]]}, 'pat_id': row[1]}
            else:
                icd[row[2]]['labels']['icd'].append(row[-1][:icd_length])
                icd[row[2]]['labels']['seq_no'].append(row[-2])
    return icd

def get_top_diagnoses(diagnoses, num_labels):
    diagnoses_list = []
    for key in diagnoses:
        icds = diagnoses[key]['labels']['icd']
        diagnoses_list.extend(icds)
    top_diagnoses = [_[0] for _ in list(Counter(diagnoses_list).most_common(num_labels))]
    return top_diagnoses

def remove_diagnoses_not_intopK(diagnoses, top_diagnoses):
    for key in diagnoses:
        icds = diagnoses[key]['labels']['icd']
        new_icds = []
        new_seq_no = []
        for i, icd in enumerate(icds):
            if icd in top_diagnoses:
                new_icds.append(icd)
                new_seq_no.append(diagnoses[key]['labels']['seq_no'][i])
        diagnoses[key]['labels']['icd'] = new_icds
        diagnoses[key]['labels']['seq_no'] = new_seq_no
    return diagnoses

def select_note_types(data, note_types):
    for d in data:
        if 'notes' in d:
            new_notes = []
            for note in d['notes']:
                if note['note_type'].lower() in note_types:
                    new_notes.append(note)
            d['notes'] = new_notes
    return data

def extract_subset_of_note(text, inc_history=True, diagnosis_short=True):
    text = text.lower()
    newtext = ''
    if inc_history:
        if 'history of present illness' in text and 'past medical history' in text:
            newtext+= text[text.index('history of present illness'):text.index('past medical history')]
        elif 'history of present illness' in text:
            start_idx = text.index('history of present illness')
            end_idx = min(text.index('history of present illness') + 1500, len(text))
            newtext += text[start_idx:end_idx]
    if 'final diagnosis' in text or 'discharge diagnosis' in text:
        if diagnosis_short:
            if 'final diagnosis' in text:
                start_idx = text.index('final diagnosis')
                end_idx = min(start_idx + 250, len(text))
                if '\n\n' in text[start_idx:]:
                    end_idx = text[start_idx:].index('\n\n')
                elif '.\n' in text[start_idx:]:
                    end_idx = text[start_idx:].index('.\n')
                newtext += text[start_idx:end_idx]
            if 'discharge diagnosis' in text:
                start_idx = text.index('discharge diagnosis')
                end_idx = min(start_idx + 250, len(text))
                if '\n\n' in text[start_idx:]:
                    end_idx = text[start_idx:].index('\n\n')
                elif '.\n' in text[start_idx:]:
                    end_idx = text[start_idx:].index('.\n')
                newtext += text[start_idx:end_idx]
        else:
            if 'final diagnosis' in text:
                newtext += text[text.index('final diagnosis'):]
            if 'discharge diagnosis' in text:
                newtext += text[text.index('discharge diagnosis'):]
    return newtext


def replace_numbers(text):
    text = re.sub('[0-9]', 'd', text)
    return text
