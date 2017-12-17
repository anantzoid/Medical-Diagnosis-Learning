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
    - Other methods of doc2vec
4. Experiments
    - Combination of different note types in learning
    - Evaluation methods
    - Sliding window etc.


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
	* Other Stats	:
		- No. of unique patient: 46520
		- No. of unique target icds: 6704
		- Top occuring:  [('4019', 16310), ('41401', 9851), ('42731', 9609), ('4280', 8830), ('25000', 6716), ('2724', 6684), ('5849', 6363), ('V053', 5680), ('51881', 5524), ('V290', 5454), ('2720', 4698), ('53081', 4672), ('5990', 4665), ('2859', 4022), ('2449', 3498), ('V3000', 3490), ('2851', 3475), ('486', 3443), ('2762', 3326), ('496', 3080)]
		- FD: (array([38983,  5160,  2096,   235,    38,     8,     0,     0,     0,     0]), array([    0,     1,     2,     5,    10,    20,    50,   100,   500,
        1000, 10000]))

- 11/14: Stats for notes per hadm_id:
	* No. of data points(hadm_id): 58976
	* Data points with no notes: 615
	* Average No. of notes per Hadm_id: 31.391
	* FD. of top occurring notes: {'Nursing/other': 821258, 'ECG': 138190, 'Radiology': 378920, 'Nursing': 220758, 'Nutrition': 9378, 'Discharge summary': 59652, 'Rehab Services': 5409, 'Consult': 98, 'Echo': 34037, 'Physician ': 140100, 'Respiratory ': 31667, 'Social Work': 2612, 'General': 8209, 'Pharmacy': 102, 'Case Management ': 954}
	* Total number of notes: 1851344
	* Average no.of tokens per notes: 241.60
	* Note with maximum tokens: 7644
	* Note with min tokens: 0
	* Unique vocab size: 2686662


	- Without stopwords:
		* Average no.of tokens per notes: 181.32
		* Note with maximum tokens: 5280
		* Unique vocab size: 2686349

- 11/16: Current Preprocessing [pipeline](src/preprocessing_pipelin.py)
	* Choose top-k labels accdn. to freq. dist.
	* Filter the data accdn. to these labels and choose top vocab list.
	* Save a filtered list of pre-trained embeddings accdn to new vocab list.

	* **Note**:
		* Might need a better tokenizer. Encountering words like patientdesatted & patientazithromycin.
		* After filtering, a note might just containe only 'unknown' tokens.
		* For label, only icd9 in seq_num 0 is taken.


- 11/25: Stats for notes per once history of present illness and discharge diagnosis has been extracted:
	* Before selection and text extraction
		* 'No. of data points:', 58976
		* 'Data points with no notes:', 615
		* 'Average No. of notes per Hadm_id:', 31.391481280520889
		* 1851344 notes, 59652 discharge notes, ratio: 3.22209162641
		* Average length of note (chars): 58733.70
		* Types of notes: {'Nursing/other': 821258, 'ECG': 138190, 'Radiology': 378920, 'Nursing': 220758, 'Nutrition': 9378, 'Discharge summary': 59652, 'Rehab Services': 5409, 'Consult': 98, 'Echo': 34037, 'Physician ': 140100, 'Respiratory ': 31667, 'Social Work': 2612, 'General': 8209, 'Pharmacy': 102, 'Case Management ': 954}
	* After selection and text extraction
		* 'No. of data points:', 58361
		* 'Data points with no notes:', 0.0
		* 'Average No. of notes per Hadm_id:', 1.0221209369270574
		* 59652 notes, 59652 discharge notes, ratio: 100.0
		* Average length of note (chars): 2047.46
		* Types of notes: {'Discharge summary': 59652}

- 11/28: Vincent added Starspace embedding initialization
	* additions to train.py
		* 3 new arguments, some logic in the middle
		* VINCENT DID NOT ACTUALLY ADD THE EMBEDDINGS INTO THE MODEL INITIALIZATION STEP
	* a new embedding_utils.py in src/
	* TO DO:
		* initialize the model with these embeddings
			* write a helper to load the embedding tsv
			* map words to indicies to embeddings tensor


Addendum

---Anant
    find avg no. of sentences
    skip alternate sentences - discourse lol
    window document

    ==> highly imbalance
        check what they do in both papers
    ==> vocab set is huge
        check what they use in both papers
    ==> windowing of sentences


    Tal et. al.
    - uses same split on mimic2
    - task is extreme multi-class classification (weston et. al.)
    - they seem to have trained some w2v with genism model (check this later in paper)
    - potential future work: exploiting discourse structure in docs.
    - Reference for type of attention being used:
        - self attentive attention https://arxiv.org/pdf/1703.03130.pdf
        - Heirarchical attention https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf


    !preprocessing paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3932472/!
        1. Run code as it is, ie, on mimic2
        2. Observe and then run on mimic3
        3. Preprocess as per Tal et. al.
        4. Implement Tal et. al model on both (if they're different) variants of the preprocessed data (in case 3. is not a complete process)

    Models with Summary only:
    1. phrase classification: seq_len batchwise
        ==> overfitting persists
    2. full summary classification
        ==> finer extraction
        ==> take top 50 labels
        ==> full summary in sequence
            *hardly loss drop*
        ==>max_seq_len: 500            
        ==>padding from behind
            *hardly loss drop*
            *overfit after few steps*
                mention of overfitting online: https://github.com/YerevaNN/mimic3-benchmarks#decompensation-prediction
        *Adding more data doesnt fight overfitting and loss doesnt drop now*
        *logs not getting saved*        

        ==> check grad norm etc. of trained model
        ==> cluster 50 labels to less no. and try

    3. with Attention
    4. sequence of notes

    PP by Perotte et al (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3932472/)
        1. A vocabulary was defined as the top 10 000 tokens ranked by their tf-idf score computed across the whole MIMIC dataset.

