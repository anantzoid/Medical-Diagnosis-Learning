#!/bin/bash

#stdbuf -oL python ../preprocess.py --data '/misc/vlgscratch2/LecunGroup/laura/medical_notes' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 4 --generatesplits 1 --numlabels 10 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile 10codesL5_UNK_content_4 --mapunk 0 2>&1 | tee datagen_10codesL5_UNK_content_4.log

stdbuf -oL python ../preprocess.py --data '/misc/vlgscratch2/LecunGroup/laura/medical_notes' --ICDcodelength 5 --notestypes 'discharge summary' --notescontent 4 --generatesplits 0 --numlabels 5 --preprocessing 'add space,remove brackets,replace numbers,replace break' --procdatafile 5codesL5_UNK_content_4 --mapunk 0 2>&1 | tee datagen_5codesL5_UNK_content_4.log
