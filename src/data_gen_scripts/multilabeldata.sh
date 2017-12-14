#!/bin/bash

# Top 50
stdbuf -oL python ../preprocess.py --data '/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 4 --generatesplits 0 --numlabels 50 --firstK 3 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile 50codesL5_UNK_content_4_top3 --mapunk 0 2>&1 | tee datagen_50codesL5_UNK_content_4_top3.log

stdbuf -oL python ../preprocess.py --data '/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 4 --generatesplits 0 --numlabels 50 --firstK 5 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile 50codesL5_UNK_content_4_top5 --mapunk 0 2>&1 | tee datagen_50codesL5_UNK_content_4_top5.log

stdbuf -oL python ../preprocess.py --data '/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 4 --generatesplits 0 --numlabels 50 --firstK 10 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile 50codesL5_UNK_content_4_top10 --mapunk 0 2>&1 | tee datagen_50codesL5_UNK_content_4_top10.log

stdbuf -oL python ../preprocess.py --data '/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 4 --generatesplits 0 --numlabels 50 --firstK 100 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile 50codesL5_UNK_content_4_top100 --mapunk 0 2>&1 | tee datagen_50codesL5_UNK_content_4_top100.log
