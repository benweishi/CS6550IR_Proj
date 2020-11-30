import gensim.models as g
import numpy as np

#parameters
model="apnews_dbow/doc2vec.bin"
doc_file="query_training_set-01.tsv"
output_vec_file="training_query_vectors.bin"
output_id_file = "training_query_id.txt"


#inference hyper-parameters
start_alpha=0.01
#infer_epoch=1000

#load model
m = g.Doc2Vec.load(model)
#test_docs = [ (x.strip().split())[1:] for x in io.open(test_docs, mode="r").readlines() ]
docs_text = []
doc_ids = []
with open(doc_file, "r") as f:
    for line in f.readlines():
        line = line.strip().split()
        id = line[0]
        text = line[1]
        doc_ids.append(id)
        docs_text.append(text)

idstring = " ".join(doc_ids)
with open(output_id_file,"w") as f:
    f.write(idstring)

#infer test vectors
print("start infering vector")

#vec = np.asarray([m.infer_vector(d, alpha=start_alpha, steps=infer_epoch) for d in docs_text],np.float32)
vec = []
for i in range(0,5):
    print(i)
    vec.append(m.infer_vector(docs_text, alpha=start_alpha))

#np.save(output_vec_file,vec)
