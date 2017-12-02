#!/bin/bash

stdbuf -oL python master_train_script.py --lr 0.001 --vocab_threshold 25 --batch_size 4 --exp_name newdata_test.pth --model_file newdata_test.pth | tee logs/test_train.log
