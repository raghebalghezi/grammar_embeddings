import gensim
import numpy as np

# gog_ana = "/Users/raghebal-ghezi/Github/Gram_Embed/node2vec/lib/python3.6/site-packages/gensim/test/test_data/questions-words.txt"

model = gensim.models.KeyedVectors.load_word2vec_format('w2v_model.bin', binary=False)

cand = ['complicate', 'failed', 'earthquakes', 'handsomely', 'responsible','The']

# for w in cand:
#     comp = model.similar_by_vector(w, topn=10)
#     # print("{}:{}\n".format(w, comp))
#     for j in comp:
#         print(f"{w}\t{j[0]}\t{j[1]}")


sim = model.n_similarity('any clear sign'.split(' '), 'the new venture'.split(' '))
sim2 = model.n_similarity('of any clearly domestic demand'.split(' '), 'in the new venture'.split(' '))
sim3 = model.n_similarity('to operate handsomely'.split(' '), 'the new venture'.split(' '))


print('-'*100)


print('any clear sign', '->' ,'the new venture', sim)
print('of any clearly domestic demand', '->' ,'in the new venture', sim2)
print('to operate handsomely', '->' ,'the new venture', sim3)

phrases = [('any clear sign', 'the venture'), 
            ('of any clearly domestic', 'in an industrial'),
            ('would need','require')]

for ph in phrases:
    ana = model.most_similar(positive=ph[0].split(' '), negative=ph[1].split(' '), topn=10)
    print("{}->{}\t{}\n".format(ph[0], ph[1], ana))

print('-'*100)


vec_dict = {}
with open('w2v_model.bin','r') as file:
    for line in file:
        row = line.strip().split(' ')
        vec_dict[row[0].lower()] = np.float32(row[1:])

def cos(v,w):
    return np.dot(v,w)/(np.linalg.norm(w) * np.linalg.norm(v))

sett = set(vec_dict.keys())

def vect(x):

    x = x.split(' ')

    if len(x) == 1:
        return vec_dict[x[0].lower()]

    v = np.sum([vec_dict[w.lower()] for w in x if w.lower() in sett], axis=0)
    return v



for w in cand:
    comp = model.similar_by_vector(w, topn=10)
    # print("{}:{}\n".format(w, comp))
    for j in comp:
        print(f"{w}\t{j[0]}\t{j[1]}\t{cos(vect(w), vect(j[0]))}")

print('*'*80)

src_sent = ['Eastern Airlines pilots were awarded $ 100 million by an arbitrator , a decision that could complicate the carrier \'s bankruptcy reorganization .',
'USX \'s profit dropped 23 % in the third quarter as improved oil results failed *-1 to offset weakness in the firm \'s steel and natural gas operations .',
'It *EXP*-1 is possible then that Santa Fe \'s real estate -- even in a state imperiled * by earthquakes -- could , one day , fetch a king \'s ransom .']

for i in zip(cand, src_sent):
    lst = [w for w in i[1].split(' ') if w in model.vocab]
    d = model.distances(i[0], lst)
    for j in zip(lst, d):
        print(i[0],j[0], cos(vect(i[0]), vect(j[0])))
    

