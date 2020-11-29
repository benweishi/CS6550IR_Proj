#!/usr/bin/python3.4
import re
import pyndri


path = "./AOL-user-ct-collection/"
filename = path + "user-ct-test-collection-0"
out = open("aol_queries-01.txt","w")
substrings = ["http", "www.", ".com", ".net", ".org", ".edu"]
queries = []
index = pyndri.Index("index")
bm25_query_env = pyndri.OkapiQueryEnvironment(index, k1=1.2, b=0.75, k3=1000)


#iterate over the lines of the first 9 files, get queries thwc at don't contain substrings and make them contain only alpharithmetic characters
#for i in range(1,11):
#    files = open(filename + str(i)+'.txt',"r")
files = open(filename + '1.txt',"r")

lines = files.readlines()
files.close()
for j in range(0,100000):
    if(j%1000==0):
        print(j)
    id = lines[j].split("\t")[0]
    query = lines[j].split("\t")[1]
    flag = 0
    for string in substrings:
        try:
            query.index(string)
            flag = 1
        except ValueError:
            continue
    if flag == 0:
        query = re.sub("[^a-zA-Z0-9 ]", '', query)
        query = " ".join(query.split())
        if (len(query) != 0):
            query_results = bm25_query_env.query(query, results_requested=10)
            if query_results == ():
                continue
            else:
                try:
                    last = query_results[9]
                except IndexError:
                    continue
                if last[1] > 0:
                    pend_query = id + '\t' + query + "\n"
                    queries.append(pend_query)

"""    
#do the same for the tenth file
files = open(path + "user-ct-test-collection-10.txt","r")
lines = files.readlines()
files.close()

for j in range(0,len(lines)):
    query = lines[j].split("\t")[1]
    flag = 0
    for string in substrings:
        try:
            query.index(string)   
            flag = 1
        except ValueError:
            continue
    if flag == 0:
        query = re.sub("[^a-zA-Z0-9 ]", '', query)
        query = " ".join(query.split())
        if (len(query) != 0):
            queries.append(query+"\n")
"""
 
#remove duplicates by using a set
queries_set = set(queries)

for i in queries_set:
    out.write(i)

out.close()
