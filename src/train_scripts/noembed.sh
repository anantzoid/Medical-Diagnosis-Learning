#!/bin/bash

stdbuf -oL python master_train_script.py --lr 0.001 --vocab_threshold 5 --batch_size 16 --num_epoch 20 --focalloss 0 --use_starspace 0 --exp_name wordsent_3.pth --model_file newdata_wordsent_3.pth | tee logs/wordsent_3.log

