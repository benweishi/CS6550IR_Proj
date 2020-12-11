import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow import keras
import os

datapath = "data/data_for_embedding_trainning"
print('Reading training query embedding data.')
# AOL_queries = pd.read_csv(os.path.join(datapath, 'query_training_set-01_10k_DEC6.tsv'), sep='\t', header=None, names=["id", "query"], dtype="string")
with open(datapath+'/training_query_id_100K_DEC10.txt') as f:
    train_queryIDList = f.readline().strip().split()
# train_qe_data = pd.read_csv(os.path.join(datapath, 'dense_vec_10k_DEC6.csv'), sep=',')
train_query_embeddings = np.load(datapath+'/trainnig_query_vec_100K_DEC10.bin.npy')
train_qe_data = pd.DataFrame(data=train_query_embeddings.reshape((100000,300)), index=train_queryIDList)

print('Reading test query embedding data.')
with open(datapath+'/test_query_id.txt') as f:
    test_queryIDList = f.readline().strip().split()
test_query_embeddings = np.load(datapath+'/test_query_vectors.bin.npy')
test_qe_data = pd.DataFrame(data=test_query_embeddings, index=test_queryIDList)

print('Reading doc embedding data.')
with open(datapath+'/doc_id.txt') as f:
    docIDList = f.readline().strip().split()
doc_embeddings = np.load(datapath+'/doc_vec.bin.npy')
doc_data = pd.DataFrame(data=doc_embeddings, index=docIDList)

print('Reading train data.')
filename = 'data/dense_vect1.csv'
df = pd.read_csv(filename)
df = df[['topic','document_name','document_score']]
df.topic = df.topic.astype(str)
y_train = df.document_score
X_train = np.c_[train_qe_data.loc[df.topic], doc_data.loc[df.document_name]]

print('Reading test data.')
filename = 'data/query.titles.csv'
df = pd.read_csv(filename)
df = df[['topic','document_name','document_score']]
df.topic = df.topic.astype(str)
y_test = df.document_score
X_test = np.c_[train_qe_data.loc[df.topic], doc_data.loc[df.document_name]]

# print('Reading training indexing data.')
# qd_index = np.load("data/q-doc_scoreIndex.bin.npy")
# qd_index_data= pd.DataFrame(qd_index, index=[qd_index[:,0],qd_index[:,1]])
# qd_index_data.columns = ["key_0","key_1","score"]
# merge_q_d_train = pd.merge(doce_data,train_qe_data,how="outer", on=[doce_data.index, test_qe_data.index])
# merge_qd_score_train =pd.merge(qd_index_data,merge_q_d_train, how = "inner", on=["key_0","key_1"])
# print(merge_qd_score_train.shape, "this value shouldn't be very small, please valid it")
# X_train = (merge_qd_score_train.drop(['key_0', 'key_1','score'], axis=1)).to_numpy()
# y_train = (merge_qd_score_train[['score']]).to_numpy()

# print('Reading test indexing data.')

# qd_index = np.load("data/q-doc_scoreIndex_test.bin.npy")
# qd__test_index_data = pd.DataFrame(qd_index, index=[qd_index[:,0],qd_index[:,1]])
# qd__test_index_data.columns = ["key_0","key_1","score"]
# merge_q_d_test = pd.merge(doce_data,test_qe_data,how="outer", on=[doce_data.index, test_qe_data.index])
# merge_qd_score_test =pd.merge(qd__test_index_data,merge_q_d_test, how = "inner", on=["key_0","key_1"])
# print(merge_qd_score_test.shape, "this value shouldn't be very small, please valid it")
# X_test = (merge_qd_score_test.drop(['key_0', 'key_1','score'], axis=1)).to_numpy()
# y_test = (merge_qd_score_test[['score']]).to_numpy()


n_features = doc_embeddings.shape[1]
print(n_features)

model = Sequential()
model.add(tf.keras.Input(shape=(n_features,)))
# model.add(tf.keras.layers.BatchNormalization())  # optional
model.add(Dense(300, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu', kernel_initializer='glorot_uniform'))
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
df['embedding_score'] = rst.flatten()
df.sort_values(['topic','embedding_score'], ascending=[True,False]).groupby('topic').head(1000)
with open("results/score_embedding.txt", "w") as f:
    r = 1
    topic = df.iloc[0].topic
    for i in range(len(X_test)):
        f.write(f'{df.iloc[i].topic} Q0 {df.iloc[i].document_name} {r} {df.iloc[i].dense_score} embedding\n')
        if topic != df.iloc[i].topic:
            r = 1
            topic = df.iloc[i].topic
        else:
            r += 1


