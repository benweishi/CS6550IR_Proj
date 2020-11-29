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

doc_vec_dic = {}
for doc_id in range(index.document_base(), index.maximum_document()):
    print(doc_id)
    s = time.time()
    terms_in_doc =  index.document(doc_id)[1]
    doc_vec = [0 for i in range(300)]

    for term_id in terms_in_doc:
        try:
            term = id2tockens[term_id]
            term_vec = list(model.wv[term])
            for i in range(300):
                doc_vec[i] += term_vec[i]
        except:
            pass
    
    if len(terms_in_doc):
        for i in range(300):
            doc_vec[i] /= len(terms_in_doc)

    doc_vec_dic[doc_id] = doc_vec
    e = time.time()
    print("time:%s"%(e-s))

with open("doc_vec", "w") as f:
    for doc_id in doc_vec_dic:
        f.write(str(doc_id))
        for idx, i in enumerate(doc_vec_dic[doc_id]):
            f.write(" dim%d:%f"%(idx, i))
        f.write("\n")
        

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


"""
print("doc processing done, w2v model done, start query processing")

f = open("query.titles.tsv", "r")
training_queries = f.readlines()
f.close()
out = open("/mnt/d/embed_test_vec.csv", "w")
headline = "topic,query"
for i in range(5):
    for j in range(300):
        headline += ",term%d_dim%d"%(i,j)
headline = headline + ",document_score,document_name"
for i in range(5):
    for j in range(300):
        headline += ",doc_first%ddim%d"%(i,j)

for i in range(5):
    for j in range(300):
        headline += ",doc_last%ddim%d"%(i,j)

out.write(headline)
out.write("\n")
l = len(training_queries)
for i in range(len(training_queries)):
    print(i)
    line =(training_queries[i].split('\n'))[0]
    topic = line.split('\t')[0]
    q = line.split('\t')[1]
    query_doc_results = bm25_query_env.query(q, results_requested=2000)
    terms = []
    dfs = []
    term_vecs = []
    qlist = q.split(' ')

    for i in range(5):
        try:
            term = qlist[i]
        except IndexError:
            term = '-100'
        terms.append(term)

        output_vec = topic + ',' + q

        try:
            term_vec = model.wv[term]
        except:
            term_vec = [0. for i in range(300)]
        term_vecs.append(term_vec)

        for v in term_vecs:
            for i in v:
                output_vec += ",%f"%(i)

    for (doc_id, score) in query_doc_results:
        BM25_score = score
        doc_name, terms_in_doc =  index.document(doc_id)
        first_5_vec = []
        last_5_vec = []
        for i in range(5):
            vec = [0 for i in range(300)]
            try:
                term = id2tockens[terms_in_doc[i]]
                #print(term)
                vec = model.wv[term]
                #print(vec)
            except:
                pass
            first_5_vec.append(vec)
        
        for i in range(5):
            vec = [0 for i in range(300)]
            try:
                term = id2tockens[terms_in_doc[-(i+1)]]
                vec = model.wv[term]
            except:
                pass
            last_5_vec.append(vec)

        vect = ""
        vect += output_vec
        vect = vect + ',' + str(BM25_score) + ',' + doc_name
        for v in first_5_vec:
            for i in v:
                vect += ",%f"%i
        for v in last_5_vec:
            for i in v:
                vect += ",%f"%i
        vect += "\n"

        out.write(vect)

out.close()
"""