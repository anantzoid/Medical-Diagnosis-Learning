## Vincent Major
## December 2017

def create_embeddings(data, sentenceembeddings, supervisedembeddings, trainpath, stsp_path = '../Starspace/starspace', embed_dim = 100):
    if sentenceembeddings == 1:
        if supervisedembeddings == 0:
            embed_name = 'embeddings/' + splitext(basename(trainpath))[0] + '_sentence_' + 'unsupervised'
            print("Beginning converting training data to sentences for UNlabeled StarSpace format.")
        else:
            embed_name = 'embeddings/' + splitext(basename(trainpath))[0] + '_sentence_' + 'supervised'
            print("Beginning converting training data to sentences for labeled StarSpace format.")
    else:
        if supervisedembeddings == 0:
            embed_name = 'embeddings/' + splitext(basename(trainpath))[0] + '_document_' + 'unsupervised'
            print("Beginning converting training data to UNlabeled StarSpace format.")
        else:
            embed_name = 'embeddings/' + splitext(basename(trainpath))[0] + '_document_' + 'supervised'
            print("Beginning converting training data to labeled StarSpace format.")
    data_name = embed_name + '_ss_train.txt'
    #stsp_data = convert_unflat_data_to_starspace_format(traindata)
    stsp_data = convert_unflat_data_to_ss_sent_vs_doc_un_supervised(data, sentenceembeddings, supervisedembeddings)
    write_starspace_format(stsp_data, data_name)
    print("Building starspace embeddings. This will take a few minutes...")
    #subprocess.call(['./Starspace/run.sh'])
    ## file written, now call starspace depending on the labeled/UNlabeled flag.
    ## placing SS model into modelfolder
    ## SS command needs to look like:
    ## ../Starspace/starspace train -trainFile <dirpath>/<trainfile>.txt -model <modelfolder>/(UN)labeled_starspace_model -trainMode 0 -label '__label__'
    if not isfile(stsp_path):
        print("Could not find the Starspace executible.")

    if supervisedembeddings == 0:
        ss_paras = [stsp_path, 'train', '-trainFile', data_name, '-model', embed_name, '-trainMode', '5', '-dim', str(embed_dim), '-normalizeText', '0']
    else:
        ss_paras = [stsp_path, 'train', '-trainFile', data_name, '-model', embed_name, '-trainMode', '0', '-dim', str(embed_dim), '-normalizeText', '0']
    print("Starspace call: ")
    print(" ".join(ss_paras))
    ss_output = subprocess.run(ss_paras, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ## print Starspace outputs
    print(ss_output.stdout.decode('utf-8'))
    print(ss_output.stderr.decode('utf-8'))
    ## Did it finish?
    last_output = " ".join(ss_output.stdout.decode("utf-8").split('\n')[-2:])
    if "Saving model in tsv format" not in last_output:
        ## won't save a file if it doesn't finish.
        print('Starspace did not complete. PANIC! \nReverting to default initialization.')
        return False
    
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
parser.add_argument('--train_path', type=str, default='data/10codesL5_UNK_content_2_top1_train_data.pkl')
parser.add_argument('--embed_dim', type=int, default=100)
parser.add_argument('--stsp_path', type=str, default='../Starspace/starspace')
parser.add_argument('--sentenceembeddings', type=int, default=1,
                    help='Whether to initialize embeddings by supplying Starspace with sentences or entire documents.')
parser.add_argument('--supervisedembeddings', type=int, default=0,
                    help='Whether to learn Starspace embeddings supervised or not.')
args = parser.parse_args()
print(args)

#torch.manual_seed(1)
#use_cuda = torch.cuda.is_available()
#if use_cuda:
#    torch.cuda.set_device(args.gpu_id)
    #torch.backends.cudnn.enabled = False


## MIMIC data code
traindata = pickle.load(open(args.train_path, 'rb'))

create_embeddings(traindata, args.sentenceembeddings, args.supervisedembeddings, args.train_path, args.stsp_path, args.embed_dim)
