#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l gpu=1,mf=16G

module load python/3.5.3
source ~/.bashrc
echo $PYTHONPATH

hostname
date
pwd

VAL='data/10codesL5_UNK_content_2_top1_valid_data.pkl'
TRAIN='data/10codesL5_UNK_content_2_top1_train_data.pkl'
MODELBASE='models/embed_exps/'
## only expecting 1 argument -- the embedding file 
EMBED=$1
## with the path, remove the path
NAME=$(basename $EMBED)
#b=$(basename $a)

EXP='embed_exps/'$NAME

## for logging 
echo $1
echo $VAL
echo $TRAIN
echo $EMBED
echo $NAME
echo $EXP
echo $MODELBASE$NAME
echo 'Beginning'

stdbuf -oL python src/master_train_script.py --train_path $TRAIN --val_path $VAL --model_file $MODELBASE$NAME --gpu_id 0 --lr 0.001 --vocab_threshold 5 --batch_size 16 --num_epoch 20 --focalloss 1 --attention 1 --exp_name $EXP'.pth' --embed_path $EMBED | tee 'logs/'$EXP
## test, epoch =1
#stdbuf -oL python src/master_train_script.py --train_path $TRAIN --val_path $VAL --model_file $MODELBASE$NAME --gpu_id 0 --lr 0.001 --vocab_threshold 5 --batch_size 16 --num_epoch 1 --focalloss 1 --attention 1 --exp_name $EXP'.pth' --embed_path $EMBED | tee 'logs/'$EXP

echo 'Done!'
