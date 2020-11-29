import pyndri
from gensim.models import KeyedVectors
import time

index = pyndri.Index("index")

token2id, id2tockens, id2df = index.get_dictionary()
del id2df

bm25_query_env = pyndri.OkapiQueryEnvironment(index, k1=1.2, b=0.75, k3=1000)


model_filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(model_filename, binary=True)
#print(model.wv['dog'], len(model.wv['dog']))

"""
with open("doc_vec") as f:
    lines = f.readlines()
    doc_vec_dic = {}
    for line in lines:
        vec = []
        t = line.split()
        for i in t[1:]:
            vec.append(float(i.split(':')[1]))
        doc_vec_dic[int(t[0])] = vec
"""

print("w2v model done, start query processing")

f = open("query.titles.tsv", "r")
training_queries = f.readlines()
f.close()
out = open("query_w2vec.csv", "w")
headline = "topic,query"
for i in range(5):
    for j in range(300):
        headline += ",term%d_dim%d"%(i,j)
for j in range(300):
    headline += ",avg_dim%d"%(j)

out.write(headline)
out.write("\n")
l = len(training_queries)
for i in range(len(training_queries)):
    print(i)
    s = time.time()
    line =(training_queries[i].split('\n'))[0]
    topic = line.split('\t')[0]
    q = line.split('\t')[1]
    #query_doc_results = bm25_query_env.query(q, results_requested=2000)
    terms = []
    dfs = []
    term_vecs = []
    qlist = q.split(' ')

    output_vec = topic + ',' + q

    avg_vecs = [0 for i in range(300)]

    for i in range(5):
        try:
            term = qlist[i]
        except IndexError:
            term = '-100'
        terms.append(term)

        try:
            term_vec = model.wv[term]
        except:
            term_vec = [0. for i in range(300)]
        term_vecs.append(term_vec)

        
    for v in term_vecs:
        for idx, i in enumerate(v):
            output_vec += ",%f"%(i)
            avg_vecs[idx] += i
    
    for i in range(len(avg_vecs)):
        avg_vecs[i] /= 5
        output_vec += ",%f"%avg_vecs[i]

    output_vec += "\n"

    e=time.time()
    print("time:%f"%(e-s))

    out.write(output_vec)

out.close()