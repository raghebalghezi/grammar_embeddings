from nltk import Tree
from main import SentGraph, Tree2graph, hierarchy_pos
from networkx.algorithms.operators.all import compose_all
import networkx as nx
import matplotlib.pyplot as plt

s1 = Tree.fromstring('(S (NP (DT The) (JJ quick) (JJ brown) (NN fox)) (VP (VBD jumped) (PP (IN over) (NP (DT The) (JJ lazy) (NN dog)))) (. .))')
s2 = Tree.fromstring('(S (NP (DT The) (NN actor)) (VP (VBD broke) (NP (PRP$ his) (NN ankle)) (SBAR (IN while) (S (VP (VBG receiving) (NP (DT The) (NN award)))))) (. .))')
s3 = Tree.fromstring('(S (NP (PRP I)) (VP (MD will) (VP (VB do) (NP (PRP$ my) (NN homework)) (NP (JJ next) (NN week)))) (. .))')

def tree2graph(t):
    lst = []
    G = nx.Graph()
    for branch in t.productions():
        leaf = str(branch).split('->')
        row = leaf[1].strip().split(' ')
        for r in row:
            lst.append((leaf[0].strip(), r.strip("'")))

    G.add_edges_from(lst)
    return G

# print(tree2graph(s1))
supergraph = compose_all([tree2graph(s) for s in [s1,s2,s3]])

options = {
    'node_color': 'green',
    'node_size': 500,
    'arrowstyle': '-|>',
    'arrowsize': 5,
}
nx.draw_networkx(supergraph, **options)
plt.show()

# nx.to_pandas_adjacency(supergraph).to_csv('adj_mat.csv')

