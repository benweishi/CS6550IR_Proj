import os
import pandas as pd
from tqdm import tqdm

infile = "data/dense_vect1.csv"
# infile = "data/query.titles.csv"
outfiel = os.path.splitext(infile)[0] + '.txt'

df = pd.read_csv(infile)
df = df.sort_values(by=['topic'])
df.document_score -= df.document_score.min()
df.document_score = df.document_score.astype(int)
X = df.iloc[:,6:]
X -= X.mean()
X /= X.std()

vali_size = len(df) // 5

def output(fn, start=0, end=None):
    with open(fn, 'w') as f:
        for i in tqdm(range(start, end)):
            x = ' '.join(f'{j+1}:{v:.3f}' for j, v in enumerate(X.loc[i]))
            f.write(f'{df.document_score[i]:.0f} qid:{df.topic[i]} {x}\n')

output(os.path.splitext(infile)[0]+'_vali.txt', 0, vali_size)
output(os.path.splitext(infile)[0]+'_train.txt', vali_size, len(df))
# output(os.path.splitext(infile)[0]+'.txt', 0, len(df))
