#!/bin/bash


hostname
date
pwd

#for FILE in $( ls data/notescontent_*_train_data.pkl ); do
## add notescontent 5
for FILE in $( ls data/notescontent_5*_train_data.pkl ); do
	echo $FILE
	for SENTENCE in 0 1; do
		echo $SENTENCE
		for SUPERVISED in 0 1; do
			echo $SUPERVISED
			echo 'Launching job'
			#qsub ss_learn_embed.sh -l mf=8G -N "ss_learn_embed"$FILE"_"$SENTENCE"_"$SUPERVISED $FILE $SENTENCE $SUPERVISED 50
			qsub ss_learn_embed.sh $FILE $SENTENCE $SUPERVISED 50 -l mf=8G -N "ss_learn_embed"$FILE"_"$SENTENCE"_"$SUPERVISED
		done
	done
done
