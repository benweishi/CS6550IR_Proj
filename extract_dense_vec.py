import pyndri
import os
from tqdm import tqdm

index_path = "data/index"

def dense_vec(queries_tsv, doc_per_query=1000, suffix='', start_query=0, max_queries=10000):
    out_file = os.path.splitext(queries_tsv)[0] + suffix + '.csv'
    index = pyndri.Index(index_path)
    token2id, id2tockens, id2df = index.get_dictionary()
    token2df = {}
    for token, id in token2id.items():
        token2df[token] = id2df[id]
    token2df['-100']=0

    id2tf = index.get_term_frequencies()
    del id2tockens
    bm25_query_env = pyndri.OkapiQueryEnvironment(index, k1=1.2, b=0.75, k3=1000)

    document_count = index.document_count()
    avg_doc_length = 0
    for doc_no in range(index.document_base(), index.maximum_document()):
        avg_doc_length += index.document_length(doc_no)
    avg_doc_length = float(avg_doc_length)/document_count

    f = open(queries_tsv, "r")
    training_queries = f.readlines()
    f.close()
    out = open(out_file, "w")
    out.write("topic,query,document_name,document_score,document_count,mean_document_length,document_length,term_frequency_1,term_frequency_2,term_frequency_3,term_frequency_4,term_frequency_5,document_frequency_1,document_frequency_2,document_frequency_3,document_frequency_4,document_frequency_5\n")
    l = len(training_queries)
    end = min(l, max_queries)
    for i in tqdm(range(start_query, end)):
        line =(training_queries[i].split('\n'))[0]
        topic = line.split('\t')[0]
        q = line.split('\t')[1]
        query_doc_results = bm25_query_env.query(q, results_requested=doc_per_query)
        terms = []
        dfs = []
        qlist = q.split(' ')

        for i in range(5):
            try:
                term = qlist[i]
            except IndexError:
                term = '-100'
            terms.append(term)
            try:
                df = token2df[term]
            except KeyError:
                df = 0
            dfs.append(df)
        for (id, score) in query_doc_results:
            BM25_score = score
            dtfs = [0,0,0,0,0]
            doc_length = index.document_length(id)
            doc_name = index.document(id)[0]
            terms_in_doc =  index.document(id)[1]
            for j in range(5):
                try:
                    term_id = token2id[terms[j]]
                except KeyError:
                    continue
                dtfs[j] = dtfs[j]+terms_in_doc.count(term_id)
            vect = topic + ',' + q + ',' + doc_name + ',' + str(BM25_score) + ',' + str(document_count) + ',' + str(avg_doc_length) + ',' +str(doc_length)+','+str(dtfs[0])+','+str(dtfs[1])+','+str(dtfs[2])+','+str(dtfs[3])+','+str(dtfs[4])+','+str(dfs[0])+','+str(dfs[1])+','+str(dfs[2])+','+str(dfs[3])+','+str(dfs[4])
            vect = vect + "\n"
            out.write(vect)
    out.close()

if __name__ == "__main__":
    dense_vec("data/query_training_set-01.tsv", 1000, suffix='_01', start_query=0)
    dense_vec("data/query_training_set-01.tsv", 1000, suffix='_02', start_query=10000)
    dense_vec("data/query.titles.tsv", 2000)