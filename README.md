Add data to `data`:
* index
* query.titles.tsv
* robust04.qrels
* aol?

## TODO
#### code
1. Check `filter_queries.py` to figure out why generating different queries. (Xinyi)
2. Run `filter_queries.py` on `AOL-01` and remove the same queries to get 100K training queries `query_training_set-01.tsv`; run `extract_dense_vec.py`(about 3h) to generate 100K training dense vector `query_training_set-01_dense_vec.csv`. (Xinyi)
3. Use the same method above to generate `query.titles.tsv` and `query.titles_dense_vec.csv`. (Xinyi)
4. Run `gen_query_w2vec.py` on data `query_training_set-01.tsv` to generate data `query_training_set-01_w2vEmbed.bin.npy` and `query_training_set-01_w2vEmbed_id.txt`. (Xinyi)
5. Use the same method above to generate `query.titles_w2vEmbed.bin.npy` and `query.titles__w2vEmbed_id.txt`.(Xinyi)
6. Try to generate Embeddings using doc2vec. (Xinyi)
7. Using above Embeddings on all the models.


#### proposal
abstract
&nbsp;1. Introduction  
&nbsp;&nbsp;&nbsp;1.1 Problem Statement    
&nbsp;&nbsp;&nbsp;1.2. Preliminary (describe all kinds of model and embeddings) (__Xinyi__: I could write the description of doc2vec embeding if we used this one)   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.2.1. Related work  
&nbsp;2. Overview  
&nbsp;&nbsp;&nbsp;2.1. Our models  
&nbsp;&nbsp;&nbsp;2.2. Our Evaluation methods(__Xinyi__).    
&nbsp;3. Experiments  
&nbsp;&nbsp;&nbsp;3.1. Dataset(__Xinyi__).    
&nbsp;&nbsp;&nbsp;3.2. Preprocess(__Xinyi__).  
&nbsp;&nbsp;&nbsp;3.3. Parameters for models(__Xinyi__).  
&nbsp;4. Summery and future work  
