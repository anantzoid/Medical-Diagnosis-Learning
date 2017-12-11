#!/bin/bash

# TOP 10
#stdbuf -oL python ../preprocess.py --data '/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 2 --generatesplits 0 --numlabels 10 --firstK 1 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile 10codesL5_UNK_content_2_top1 --mapunk 0 2>&1 | tee datagen_10codesL5_UNK_content_2_top1.log

stdbuf -oL python ../preprocess.py --data '/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 3 --generatesplits 0 --numlabels 10 --firstK 1 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile 10codesL5_UNK_content_3_top1 --mapunk 0 2>&1 | tee datagen_10codesL5_UNK_content_3_top1.log

stdbuf -oL python ../preprocess.py --data '/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 4 --generatesplits 0 --numlabels 10 --firstK 1 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile 10codesL5_UNK_content_4_top1 --mapunk 0 2>&1 | tee datagen_10codesL5_UNK_content_4_top1.log

#stdbuf -oL python ../preprocess.py --data '/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 5 --generatesplits 0 --numlabels 10 --firstK 1 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile 10codesL5_UNK_content_5_top1 --mapunk 0 2>&1 | tee datagen_10codesL5_UNK_content_5_top1.log


# Top 50
stdbuf -oL python ../preprocess.py --data '/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 2 --generatesplits 0 --numlabels 50 --firstK 1 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile 50codesL5_UNK_content_2_top1 --mapunk 0 2>&1 | tee datagen_50codesL5_UNK_content_2_top1.log

stdbuf -oL python ../preprocess.py --data '/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 3 --generatesplits 0 --numlabels 50 --firstK 1 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile 50codesL5_UNK_content_3_top1 --mapunk 0 2>&1 | tee datagen_50codesL5_UNK_content_3_top1.log

stdbuf -oL python ../preprocess.py --data '/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 4 --generatesplits 0 --numlabels 50 --firstK 1 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile 50codesL5_UNK_content_4_top1 --mapunk 0 2>&1 | tee datagen_50codesL5_UNK_content_4_top1.log

stdbuf -oL python ../preprocess.py --data '/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 5 --generatesplits 0 --numlabels 50 --firstK 1 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile 50codesL5_UNK_content_5_top1 --mapunk 0 2>&1 | tee datagen_50codesL5_UNK_content_5_top1.log


# Whole note
stdbuf -oL python ../preprocess.py --data '/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 1 --generatesplits 0 --numlabels 10 --firstK 1 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile 10codesL5_UNK_content_1_top1 --mapunk 0 2>&1 | tee datagen_10codesL5_UNK_content_1_top1.log

stdbuf -oL python ../preprocess.py --data '/misc/vlgscratch2/LecunGroup/laura/medical_notes/processed_data' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 1 --generatesplits 0 --numlabels 50 --firstK 1 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile 50codesL5_UNK_content_1_top1 --mapunk 0 2>&1 | tee datagen_50codesL5_UNK_content_1_top1.log
