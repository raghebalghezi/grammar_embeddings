from main import SentGraph, Tree2graph, hierarchy_pos
import matplotlib.pyplot as plt
from graph import Node2Vec
import networkx as nx
from networkx.algorithms.operators.all import compose_all, intersection_all, disjoint_union_all, union_all
import nltk
import numpy as np

def cos(v,w):
    return np.dot(v,w)/(np.linalg.norm(w) * np.linalg.norm(v))


ptk = "/Users/raghebal-ghezi/nltk_data/corpora/ptb/WSJ/merged.mrg"

lst_graph = []
for tree in nltk.corpus.treebank.parsed_sents(ptk)[:1000]:
    t = Tree2graph(str(tree))
    lst_graph.append(t.graph)


multigraph = compose_all(lst_graph)
# nx.draw(multigraph, with_labels=True)
# plt.show()
# nx.to_pandas_dataframe(multigraph).to_csv('adj_mat.csv')
dist_query = dict(nx.all_pairs_shortest_path_length(multigraph)) #dist


print("No of edges", multigraph.number_of_edges())
w2vparams={"window":30, "size":20, "negative":30, "iter":10, "batch_words":128}#, "walklen":20}

n2v = Node2Vec(threads=0, w2vparams=w2vparams)
# print(n2v.w2vparams)
n2v.fit(multigraph, verbose=True)

# print(n2v.predict('month'))
print(n2v.model.wv.most_similar('politically', topn=10))

print(n2v.model.wv.similarity("Bush", "administration"), dist_query["Bush"]["administration"]) #sisters
print(n2v.model.wv.similarity("business", "executives"), dist_query["business"]["executives"]) #sisters
print(n2v.model.wv.similarity("health", "issues"), dist_query["health"]["issues"]) #sisters
print(n2v.model.wv.similarity("spending", "more"), dist_query["spending"]["more"]) #parent-child
print(n2v.model.wv.similarity("spending", "time"), dist_query["spending"]["time"]) #parent-child
print(n2v.model.wv.similarity("will", "feature"), dist_query["will"]["feature"]) #parent-child
print(n2v.model.wv.similarity("eroding", "strength"), dist_query["eroding"]["strength"]) #grandparent-child
print(n2v.model.wv.similarity("dealing", "Democrats"), dist_query["dealing"]["Democrats"]) #grandparent-child
print(n2v.model.wv.similarity("complained", "Democrats"), dist_query["complained"]["Democrats"]) #grandparent-child
print(n2v.model.wv.similarity("will", "Democrats"), dist_query["will"]["Democrats"])
# print(n2v.model.wv.most_similar(positive=['medical', 'costs'], negative=['companies']))
# n2v.model.predict_output_word('Labor is upset because many companies'.split(' '), topn=-10)
# complicate = n2v.predict('democratic')
# make = n2v.predict('demands')

# print(cos(make, complicate), cos(np.ones(3),np.ones(3)))

n2v.model.wv.save_word2vec_format('w2v_model_new.bin', )