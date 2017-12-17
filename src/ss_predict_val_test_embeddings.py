## Vincent Major
## December 2017

def test_embeddings(vdata, tdata, embed_name, stsp_path = '../Starspace/starspace'):
    ## first, detect if embeddings are sentence or document
    if 'sentence' in embed_name:
        sentenceembeddings = 1
    else: 
        sentenceembeddings = 0
    
    ## second, convert val and test to ss format
    #stsp_data = convert_unflat_data_to_starspace_format(traindata)
    stsp_data_v = convert_unflat_data_to_ss_sent_vs_doc_un_supervised(vdata, sentenceembeddings, 1)
    stsp_data_t = convert_unflat_data_to_ss_sent_vs_doc_un_supervised(tdata, sentenceembeddings, 1)
    
    val_data_name = embed_name + '_val.txt'
    val_out_name = embed_name + '_val_output.txt'
    test_data_name = embed_name + '_test.txt'
    test_out_name = embed_name + '_test_output.txt'
    
    write_starspace_format(stsp_data_v, val_data_name)
    write_starspace_format(stsp_data_t, test_data_name)
    print("Done writing val and test files.")
    
    print("Building starspace embeddings. This will take a few minutes...")
    #subprocess.call(['./Starspace/run.sh'])
    ## file written, now call starspace depending on the labeled/UNlabeled flag.
    ## placing SS model into modelfolder
    ## SS command needs to look like:
    ## ../Starspace/starspace train -trainFile <dirpath>/<trainfile>.txt -model <modelfolder>/(UN)labeled_starspace_model -trainMode 0 -label '__label__'
    if not isfile(stsp_path):
        print("Could not find the Starspace executible.")

    #ss_paras = [stsp_path, 'train', '-trainFile', data_name, '-model', embed_name, '-trainMode', '5', '-dim', str(embed_dim), '-normalizeText', '0']
    val_ss_paras = [stsp_path, 'test', '-testFile', val_data_name, '-model', embed_name, '-predictionFile', val_out_name, '-K', '1']
    test_ss_paras = [stsp_path, 'test', '-testFile', test_data_name, '-model', embed_name, '-predictionFile', test_out_name, '-K', '1']
    
    print("Starspace call: ")
    print(" ".join(val_ss_paras))
    print(" ".join(test_ss_paras))
    val_ss_output = subprocess.run(val_ss_paras, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    test_ss_output = subprocess.run(test_ss_paras, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ## print Starspace outputs
    print('Val file')
    print(val_ss_output.stdout.decode('utf-8'))
    print(val_ss_output.stderr.decode('utf-8'))
    print('Test file')
    print(test_ss_output.stdout.decode('utf-8'))
    print(test_ss_output.stderr.decode('utf-8'))
    ## Did it finish?
    
    return True


import pickle
from os.path import basename, splitext, isfile
#import os
import subprocess
import torch
#from attention_databuilder import *
#from attention_models import *
from embedding_utils import *
#from evaluate import *
#from loss import *
import argparse
parser = argparse.ArgumentParser(description='MIMIC III note embeddingss data preparation')
parser.add_argument('--val_path', type=str, default='data/10codesL5_UNK_content_2_top1_valid_data.pkl')
parser.add_argument('--test_path', type=str, default='data/10codesL5_UNK_content_2_top1_test_data.pkl')
parser.add_argument('--embed_path', type=str, default='embeddings/test.tsv')
parser.add_argument('--stsp_path', type=str, default='../Starspace/starspace')
args = parser.parse_args()
print(args)


## MIMIC data code
valdata = pickle.load(open(args.val_path, 'rb'))
testdata = pickle.load(open(args.test_path, 'rb'))

test_embeddings(valdata, testdata, args.embed_path, stsp_path=args.stsp_path)
