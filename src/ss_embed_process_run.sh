#!/bin/bash
#$ -S /bin/bash
#$ -cwd

module load python/3.5.3
source ~/.bashrc
echo $PYTHONPATH

hostname
date
pwd

## known to work
#stdbuf -oL python ../preprocess.py --data '/misc/vlgscratch2/LecunGroup/laura/medical_notes' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 2 --generatesplits 0 --numlabels 10 --firstK 1 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile 10codesL5_UNK_content_2_top1 --mapunk 0 2>&1 | tee datagen_10codesL5_UNK_content_2_top1.log
## same for vincent
#stdbuf -oL python ../preprocess.py --data '/ifs/home/vjm261/Medical-Diagnosis-Learning/data' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 2 --generatesplits 0 --numlabels 10 --firstK 1 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile 10codesL5_UNK_content_2_top1 --mapunk 0 2>&1 | tee datagen_10codesL5_UNK_content_2_top1.log

echo "Starting!"
stdbuf -oL python src/preprocess.py --data '/ifs/home/vjm261/Medical-Diagnosis-Learning/data' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 1 --generatesplits 0 --numlabels 10 --firstK 1 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile ../embeddings/data/notescontent_1 --mapunk 0 2>&1 | tee embeddings/datagen_notescontent_1.log
stdbuf -oL python src/preprocess.py --data '/ifs/home/vjm261/Medical-Diagnosis-Learning/data' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 2 --generatesplits 0 --numlabels 10 --firstK 1 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile ../embeddings/data/notescontent_2 --mapunk 0 2>&1 | tee embeddings/datagen_notescontent_2.log
stdbuf -oL python src/preprocess.py --data '/ifs/home/vjm261/Medical-Diagnosis-Learning/data' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 3 --generatesplits 0 --numlabels 10 --firstK 1 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile ../embeddings/data/notescontent_3 --mapunk 0 2>&1 | tee embeddings/datagen_notescontent_3.log
stdbuf -oL python src/preprocess.py --data '/ifs/home/vjm261/Medical-Diagnosis-Learning/data' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 4 --generatesplits 0 --numlabels 10 --firstK 1 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile ../embeddings/data/notescontent_4 --mapunk 0 2>&1 | tee embeddings/datagen_notescontent_4.log
stdbuf -oL python src/preprocess.py --data '/ifs/home/vjm261/Medical-Diagnosis-Learning/data' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 5 --generatesplits 0 --numlabels 10 --firstK 1 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile ../embeddings/data/notescontent_5 --mapunk 0 2>&1 | tee embeddings/datagen_notescontent_5.log
echo "Done!"
