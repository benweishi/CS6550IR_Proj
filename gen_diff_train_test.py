#!/usr/bin/python3

files = open("aol_queries-01.txt","r")
training_queries = []
ids = []
for r in files.readlines():
    ids.append(r.split('\t')[0])
    training_queries.append(r.split('\t')[1])
files.close()

validation_queries = []
files = open("query.titles.tsv","r")
for r in files.readlines():
    validation_queries.append(r.split('\t')[1])
files.close()
out = open("query_training_set-01.tsv","w")

for i in range(0,len(training_queries)):
    if training_queries[i] not in validation_queries:
        out.write(ids[i]+'\t'+training_queries[i])
    else:
        print("there is a same pair",training_queries[i])

out.close()
