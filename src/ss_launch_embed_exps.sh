#!/bin/bash

## script to launch a GPU job for each set of embeddings
hostname
date
pwd

for EMBED in $( ls embeddings/notes*.tsv ); do
	echo $EMBED
	echo 'Launching job'
	qsub src/ss_train_embed.sh $EMBED -l gpu=1,mf=16G -N "ss_train_embed"$EMBED
done
echo 'Done!'
