import pickle
import argparse
import numpy as np
from attention_databuilder import *
PADDING = "<PAD>"
UNKNOWN = "UNK"

def count_unks(data):
    unks = 0
    toks = 0
    for elem in data:
        for sent in elem[1]:
            unks += sum([1 for _ in sent if _ == 'UNK'])
            toks += len(sent)
    print("Toks: {} Unks: {} % UNKs: {:.3f}".format(toks, unks, unks / float(toks)))


def get_stats(data):
    words_per_sent = []
    sents_per_note = []
    for elem in data:
        sents_per_note.append(len(elem[1]))
        for sent in elem[1]:
            words_per_sent.append(len(sent))
    print("Avg sents: {:.3f}, Min sents: {}: Max sents: {}".format(
          np.mean(sents_per_note), min(sents_per_note), max(sents_per_note)))
    print("Avg words: {:.3f}, Min words: {}: Max words: {}".format(
          np.mean(words_per_sent), min(words_per_sent), max(words_per_sent)))


parser = argparse.ArgumentParser(
    description='MIMIC III notes data preparation')
parser.add_argument('--train_path', type=str,
                    default='/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data/10codesL5_UNK_content_2_top1_train_data.pkl')
parser.add_argument('--val_path', type=str,
                    default='/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data/10codesL5_UNK_content_2_top1_valid_data.pkl')
parser.add_argument('--vocab_threshold', type=int,
                    default=5)
args = parser.parse_args()
print(args)

traindata = pickle.load(open(args.train_path, 'r'))
valdata = pickle.load(open(args.val_path, 'r'))
print("Train size:", len(traindata))
print("Valid size:", len(valdata))

label_map = {i: _ for _, i in enumerate(get_labels(traindata))}
vocabulary, token2idx = build_vocab(
    traindata, PADDING, UNKNOWN, args.vocab_threshold)
print("Vocab size:", len(vocabulary))

print("Label mix training data")
count_labels(traindata)
print()
print("Label mix valid data")
count_labels(valdata)

print("training data stats")
get_stats(traindata)
count_unks(traindata)
print()

print("valid data stats")
get_stats(valdata)
count_unks(valdata)
print()
