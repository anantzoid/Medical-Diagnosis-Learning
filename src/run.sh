#!/bin/bash

#stdbuf -oL python master_train_script.py --lr 0.001 --vocab_threshold 5 --batch_size 16 --num_epoch 20 --exp_name newdata_test.pth --model_file newdata_test.pth | tee logs/test_train.log

stdbuf -oL python master_train_script.py --lr 0.001 --vocab_threshold 5 --batch_size 16 --num_epoch 20 --attention 1 --exp_name attention_1.pth --model_file newdata_attention_1.pth | tee logs/attention_1.log

