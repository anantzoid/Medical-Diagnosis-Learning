### Experiments

~~David Sontag's ICD9 embeddings. Take sum, avg, (min, max) or embeddings per encounter, then feed into LSTM, BiLSTM~~

Objective: Use all the notes created during a each admission (unqiue hadm_id) to predict the diagnosis_icd of that admission:

1. Sum/avg of pubmed embeddings of each note and LSTM model on those for chosen ccs diagnosis categories. (Anant)
	- Generate data dump clustering notes hadm_id-wise sorted by timestamp.
2. Train word embeddings using word2vec/Starspace to use in 1. (Vincent)
	- Notes' Stats
3. Explore methods to train 1 & 2 in better ways
	- Attention
	- Multiple embeddings per notes
	- [learning heirarchial representation](https://arxiv.org/pdf/1705.08039.pdf)
	- Learning from convolutions (multiple filters to generate a sequence)
	- Maybe try grounding?


#### Workplan to deadline

- ~~11/12: Literature review notes~~ Re-evaluate problem after running 1. (See update log)
- 11/16: #1.
- 11/19: Data prep code, Baselines (1 - 4)
- 11/26: Better note representations
- 12/03: Better note representations
- 12/10:
- 12/17 (due): Paper writeup

### Update Log
- 11/13: MIMIC3 encounters per patient is confined to 2 encounter on average. Direct mapping of ICD9 codes in Exp. 1 didn't help the network in learning much. 
Issues:
	* <pad> and <unk> tokens weren't there in the pretrained-set
	* There was too much padding in each sequence since max_seq_length in each batch would be around 10 or something.
