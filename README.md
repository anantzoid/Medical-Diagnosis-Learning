# Medical-Diagnosis-Learning## Frequency Distribution of number of encounters per patient(Unique combination of (pat\_id, hadm\_id) from diagnosis table)
1-2: 1057<br>2-5: 9584<br>5-10: 23368<br>10-20: 19214<br>20-30: 4596<br>30-50: 736<br>**Data Prepartion Technique** according to amiajnl-2013-002159.pdf (used by the GRNN paper as well). + Windowing (also used in Dubois et.al.)Questions to ask: What should be the objective function in training word embeddings of each note, such that, on their aggregation* to build note embeddings conditioned on structural data, that specific information is preserved distinctly that would be helpful in predicting the diagnoses. 
  *Aggregation would involve min, mean, max of embeddings (Dubois et. al.)Preliminary Model:
![model0](images/model0.jpg)

###Training word embeddings:
- Methods described in Sontag et. al.<br>
- Take pretrained embeddings(fasttext) and just train our required OOV tokens (not sure how this is done. Related post in Piazza).<br>
- Use [StarSpace](https://github.com/facebookresearch/StarSpace) to train embeddings from scratch with the objective of classifying diagnoses for each note. Starspace is very recent and don't think medical notes have been trained on it. So not sure if it'll work. But it's made by Jason Weston, so if any questions regarding this, Abhinav can relay our queries to him.

Also, if we don't want the word embeddings to be a bottleneck in the subsequent pipeline (in the figure above), we can download pretrained embeddings of diagnosis codes ([Sontag et. al.](https://github.com/clinicalml/embeddings)) and just use them instead of notes embeddings for now to continue building the attention based prediction model.