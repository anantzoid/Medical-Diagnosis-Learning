#!/bin/bash
#$ -S /bin/bash
#$ -cwd

module load python/3.5.3
source ~/.bashrc
echo $PYTHONPATH

hostname
date
pwd

## expecting only one argument embedding path
#echo $1
## hard coded

#python src/learn_embeddings.py --train_path $1 --embed_dim $4 --stsp_path Starspace/starspace --sentenceembeddings $2 --supervisedembeddings $3
#python src/test_val_test_embeddings.py --embed_path $1 --stsp_path Starspace/starspace

## doc then sent
python src/test_val_test_embeddings.py --embed_path embeddings/notescontent_2_train_data_document_supervised.tsv --stsp_path Starspace/starspace
python src/test_val_test_embeddings.py --embed_path embeddings/notescontent_2_train_data_sentence_supervised.tsv --stsp_path Starspace/starspace
