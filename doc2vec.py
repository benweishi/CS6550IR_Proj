import time

class Docs:
    def __init__(self, filename="data/robust04.txt"):
        with open(filename) as f:
            self.trec = f.read()
        self.start = 0
        print(len(self.trec))

    def __iter__(self):
        return self

    def __next__(self):
        # print(self.start)
        if self.start > len(self.trec)-10:
            raise StopIteration
        else:
            docno = self.contentOfNext('DOCNO')
            text = self.contentOfNext('TEXT')
            return docno, text
    
    def contentOfNext(self, tag):
        start = self.afterNext(tag)
        end = self.beforeNext(tag)
        return self.trec[start:end]
    
    def afterNext(self, tag):
        pos = self.trec.find('<'+tag+'>', self.start) + len(tag)+ 3
        self.start = pos
        return pos
    
    def beforeNext(self, tag):
        pos = self.trec.find('</'+tag+'>', self.start) -1
        self.start = pos + len(tag) + 4
        return pos

start = time.time()
docs = Docs()
print(f'Readed in {time.time() - start}')
start = time.time()
i = 0
for d, _ in docs:
    i += 1
    if (i%10000 == 0):
        print(i, d)

print(f'Iterated in {time.time() - start}')
