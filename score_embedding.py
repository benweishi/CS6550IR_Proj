import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow import keras

queryIDList = []
with open('data/query.titles.tsv') as f:
    for cnt, line in enumerate(f):
        id = line.split('\t')[0]
        queryIDList.append(id)

n = len(queryIDList)
d = 5
embeddings = np.load('data/query_vectors.bin.npy')
print(embeddings.shape)
df = pd.DataFrame(data=embeddings, index=queryIDList)
# print(df)
# print(df.loc[['302','305','302']])
# print(df.loc[['302','305']].to_numpy())
'''
def input_fn(data_set, mean=None, std=None):
    labels = data_set['document_score'].values
    FEATURES = ['mean_document_length', 'document_length', 'term_frequency_1', 'term_frequency_2',
                'term_frequency_3', 'term_frequency_4', 'term_frequency_5', 'document_frequency_1',
                'document_frequency_2','document_frequency_3', 'document_frequency_4', 'document_frequency_5']
    feature_cols = data_set[FEATURES]

    return feature_cols, labels

print('Reading train data.')
filename = 'data/dense_vect1.csv'
df = pd.read_csv(filename)
X_train, y_train = input_fn(df)

print('Reading test data.')
filename = 'data/query.titles.csv'
df = pd.read_csv(filename)
X_test, y_test = input_fn(df)

n_features = X_train.shape[1]

model = Sequential()
model.add(tf.keras.Input(shape=(n_features,)))
# model.add(tf.keras.layers.BatchNormalization())  # optional
model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dropout(0.5))
model.add(Dense(1))

opt = keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=opt, loss='mse')

model.fit(X_train, y_train, epochs=2, batch_size=512, verbose=1, validation_split=0.2)

rst = model.predict(X_test)
print(rst, rst.max(), rst.min())

# output, row format:
# topic 'Q0' document_name rank dense_score 'dense_score'
# example row:
# 301 Q0 FBIS4-41991 1 8.130586624145508 dense_score
df['dense_score'] = rst.flatten()
df.sort_values(['topic','dense_score'], ascending=[True,False]).groupby('topic').head(1000)
with open("results/score_dense.txt", "w") as f:
    r = 1
    topic = df.iloc[0].topic
    for i in range(len(X_test)):
        f.write(f'{df.iloc[i].topic} Q0 {df.iloc[i].document_name} {r} {df.iloc[i].dense_score} dense_score\n')
        if topic != df.iloc[i].topic:
            r = 1
            topic = df.iloc[i].topic
        else:
            r += 1
'''