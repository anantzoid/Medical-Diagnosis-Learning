#!/bin/bash

# History of present illness, discharge diagnosis, final diagnosis
stdbuf -oL python ../preprocess.py --data '/misc/vlgscratch2/LecunGroup/laura/medical_notes' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 4 --generatesplits 0 --numlabels 50 --firstK 1 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile 50codesL5_UNK_content_4_top1 --mapunk 0 2>&1 | tee datagen_50codesL5_UNK_content_4_top1.log

#stdbuf -oL python ../preprocess.py --data '/misc/vlgscratch2/LecunGroup/laura/medical_notes' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 4 --generatesplits 0 --numlabels 50 --firstK 2 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile 50codesL5_UNK_content_4_top2 --mapunk 0 2>&1 | tee datagen_50codesL5_UNK_content_4_top2.log

#stdbuf -oL python ../preprocess.py --data '/misc/vlgscratch2/LecunGroup/laura/medical_notes' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 4 --generatesplits 0 --numlabels 50 --firstK 3 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile 50codesL5_UNK_content_4_top3 --mapunk 0 2>&1 | tee datagen_50codesL5_UNK_content_4_top3.log

# All note
#stdbuf -oL python ../preprocess.py --data '/misc/vlgscratch2/LecunGroup/laura/medical_notes' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 1 --generatesplits 0 --numlabels 50 --firstK 1 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile 50codesL5_UNK_content_1_top1 --mapunk 0 2>&1 | tee datagen_50codesL5_UNK_content_1_top1.log

#stdbuf -oL python ../preprocess.py --data '/misc/vlgscratch2/LecunGroup/laura/medical_notes' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 1 --generatesplits 0 --numlabels 50 --firstK 2 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile 50codesL5_UNK_content_1_top2 --mapunk 0 2>&1 | tee datagen_50codesL5_UNK_content_1_top2.log

#stdbuf -oL python ../preprocess.py --data '/misc/vlgscratch2/LecunGroup/laura/medical_notes' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 1 --generatesplits 0 --numlabels 50 --firstK 3 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile 50codesL5_UNK_content_1_top3 --mapunk 0 2>&1 | tee datagen_50codesL5_UNK_content_1_top3.log
