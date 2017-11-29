from collections import Counter
import numpy as np
import torch
import torch.utils.data as data

ENGLISH_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "nobody", "noone",
    "nor", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves", ":", ",", "?", ""])

class FlatData(data.Dataset):
    def __init__(self, data, word_2_idx, label_map):
        super(FlatData, self).__init__()
        for i, row in enumerate(data):
            hadm_id = row[0]
            text = [word_2_idx[word] for sent in row[1] for word in sent]
            words = [word for sent in row[1] for word in sent] ## for embeddings
            label = [label_map[l] for l in row[2].split(' ')]
            label_onehot = np.zeros(len(label_map.keys()))
            for l in label:
                label_onehot[l] = 1
            if i == 0:
                print("Example encoded data...")
                print(hadm_id, text, label, label_onehot)
                print()
            data[i] = {}
            data[i]['text_index_sequence'] = text
            data[i]['label'] = label_onehot
            data[i]['id'] = hadm_id
            data[i]['words'] = words ## for embeddings
            data[i]['dx_index'] = row[2].split(' ') ## for embeddings
        self.data = data

    def __getitem__(self, index):
        return (self.data[index]['text_index_sequence'], self.data[index]['label'])

    def __len__(self):
        return len(self.data)
    
    def get_dx_index(self, index):
        return self.data[index]['dx_index']
        
    def get_words(self, index):
        return self.data[index]['words']


def flat_batch_collate(batch):
    max_note_len = max([len(_[0]) for _ in batch])
    x = torch.zeros(len(batch), max_note_len)
    y = []
    for i, example in enumerate(batch):
        y.append(np.asarray(example[1]))
        for j, word in enumerate(example[0]):
            x[i, j] = float(word)
    y = np.stack(y)
    # print(batch[0][0][50:55])
    # print(x[0][50:55])
    # print(batch[0][1])
    # print(y[0])
    # print(batch[8][0][50:55])
    # print(x[8][50:55])
    # print(batch[8][1])
    # print(y[8])
    return (x.long(), torch.from_numpy(y))


def build_dictionary(train_data, PADDING):
    vocab = []
    for d in train_data:
        note = d[1]
        for sent in note:
            vocab.extend(sent)
    vocab = [_[0] for _  in list(Counter(vocab).most_common())]
    try:
      print("Deleting UNK")
      idx = vocab.index('UNK')
      del vocab[idx]
    except:
      print("No UNK found")
    vocab = [PADDING, 'UNK'] + vocab
    print("Size of vocabulary: {}".format(len(vocab)))
    print("Top 100 words...")
    print(vocab[:100])
    print()
    word_indices = dict(zip(vocab, range(len(vocab))))
    return (word_indices, vocab)


def build_label_map(labels):
    label_indices = dict(zip(labels, range(len(labels))))
    return label_indices

def remove_stopwords(data):
    avg_length_before = 0
    avg_length_after = 0
    max_b = 0
    max_a = 0
    for d in data:
      note = d[1]
      n = [[tok for tok in sent if tok not in ENGLISH_STOP_WORDS] for sent in note]
      len_b = sum([1 for sent in note for tok in sent])
      len_a = sum([1 for sent in n for tok in sent])
      avg_length_before += len_b
      avg_length_after += len_a
      if len_b > max_b:
        max_b = len_b
      if len_a > max_a:
        max_a = len_a
      d[1] = n
    print("Avg length before: {}, avg length after removing stopwords: {}".format(
        avg_length_before / float(len(data)), avg_length_after / float(len(data))))
    print("Max length before: {}, max length after removing stopwords: {}".format(
       max_b, max_a)) 
    return data
