from build_datasets_utils import *
import numpy as np
import sys
import torch

def load_starspace_embeds(filepath, embed_dim):
    lines = [line.rstrip('\n').split('\t') for line in open(filepath, 'r')]
    print(len(lines))
    print(len(lines[0]))
    embed_dict = {}
    for line in lines:
        word = line[0]
        embedding = np.array([float(dim) for dim in line[1:]])
        # print(word)
        # print(embedding)
        # print(embedding.shape)
        # sys.exit()
        if embed_dim != embedding.shape[0]:
            print("Error, wrong dimension embedding")
        embed_dict[word] = embedding
    return embed_dict


def create_starspace_embedding_matrix(stsp_embed, token2idx, vocab_size, embed_dim):
    embedding_matrix = torch.FloatTensor(vocab_size, embed_dim)
    print("Embedding matrix size {}".format(embedding_matrix.size()))
    init_range = 0.1
    embedding_matrix.uniform_(-init_range, init_range)
    init_embed = 0
    for tok in token2idx:
        if tok in stsp_embed:
            embedding_matrix[token2idx[tok], :].copy_(torch.from_numpy(stsp_embed[tok]))
            init_embed += 1
    print("{} embeddings of a total of {} initialized with starspace embeddings".format(
        init_embed, vocab_size - 2))
    return embedding_matrix


def convert_flatdata_to_starspace_format(data, flag_labeled = True, label_prefix = "__label__"):
    ## expecting data after converted by build_datasets_utils.FlatData
    ## dict_keys(['text_index_sequence', 'label', 'id', 'text'])
    ## where 'label' is actually an index.

    output = []
    if flag_labeled == True:
        for i in range(len(data)):
            ## add label_prefix (__label__) to each label and add to list of words.
            ## join by space
            output.append(" ".join([label_prefix + d for d in data.get_dx_index(i)] + data.get_words(i)))
        print("Finished converting labeled data to StarSpace format.")
        return output
        ## Done
    elif flag_labeled == False:
        for i, rec in enumerate(data):
            ## ONLY list of words.
            ## join by space
            output.append(" ".join(data.get_words(i)))
        print("Finished converting UNlabeled data to StarSpace format.")
        return output

def convert_unflat_data_to_starspace_format(data):
    output = []
    for i, rec in enumerate(data):
        note = rec[1]
        for sent in note:
            output.append(" ".join([word for word in sent]))
        if i == 0:
            print(note)
            print(output)
    print("Finished converting UNlabeled data to StarSpace format.")
    return output

def write_starspace_format(records, filepath):
    with open(filepath, 'w') as file_handler:
            for record in records:
                if record != '':
                    file_handler.write("{}\n".format(record))
    print("Finished writing StarSpace data to file.")
