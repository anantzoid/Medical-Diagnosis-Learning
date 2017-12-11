#!/bin/bash
#$ -S /bin/bash
#$ -cwd

module load python/3.5.3
source ~/.bashrc
echo $PYTHONPATH

hostname
date
pwd

echo $1
echo $2
echo $3
echo $4

python src/learn_embeddings.py --train_path $1 --embed_dim $4 --stsp_path Starspace/starspace --sentenceembeddings $2 --supervisedembeddings $3

