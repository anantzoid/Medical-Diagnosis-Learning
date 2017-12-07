#!/bin/bash

#stdbuf -oL python master_train_script.py --lr 0.001 --vocab_threshold 5 --batch_size 16 --num_epoch 20 --exp_name newdata_test.pth --model_file newdata_test.pth | tee logs/test_train.log

stdbuf -oL python master_train_script.py --lr 0.001 --vocab_threshold 5 --batch_size 16 --num_epoch 20 --attention 0 --cbow 1 --exp_name cbow_1.pth --model_file newdata_cbow_1.pth | tee logs/cbow_1.log

