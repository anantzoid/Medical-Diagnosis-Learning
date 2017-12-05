#!/bin/bash

stdbuf -oL python master_train_script.py --lr 0.001 --vocab_threshold 5 --batch_size 16 --num_epoch 20 --exp_name newdata_test.pth --model_file newdata_test.pth | tee logs/test_train.log
#stdbuf -oL python master_train_script.py --lr 0.001 --vocab_threshold 5 --batch_size 32 --num_epoch 20 --exp_name newdata_test.pth --model_file newdata_test.pth | tee logs/test_train.log

# stdbuf -oL python master_train_script.py --lr 0.001 --vocab_threshold 5 --batch_size 4 --exp_name newdata_test.pth --model_file newdata_test.pth --build_starspace 1 | tee logs/test_train.log
