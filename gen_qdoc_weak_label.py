
import numpy as np

dense_file = "../query.titles.csv"
pairs = []
#mytype = 'S10,S10,float32'
with open(dense_file) as f:
    next(f)
    for row in f:
        rowlist = row.strip().split( ',')
        query_id = rowlist[1]
        doc_id = rowlist[3]
        score = rowlist[4]
        idx = str(query_id)+'-'+str(doc_id)
        pairs.append([str(query_id), str(doc_id),score])
print(pairs[1][1])
#output = np.array(pairs,dtype=mytype)
output = np.array(pairs)
print(output)

np.save("q-doc_scoreIndex_test.bin",output)

