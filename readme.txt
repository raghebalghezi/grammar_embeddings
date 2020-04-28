
gramEmbed.bin is a gensim-compliant grammatical embeddings pretrained on PennTree WSJ corpus.

graph.py contains implementation of Node2vec algorithm (Not Mine)
main.py contains helper methods to convert Consit. Parse tree into Networkx graph
model.py pretraining procedure of grammatical embeddings (Requires Penntree WSJ corpus!)


To reproduce the evaluation results, run:

wget http://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip
pip install requirements.txt

Use:

1. pos_tag.py to see results of SVM
2. pos_tag_crf.py to see results of CRF
3. task.py to see results from 3.1 to 3.3 in the report.


