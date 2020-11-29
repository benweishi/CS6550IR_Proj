import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow import keras

def input_fn(data_set):
    labels = data_set['document_score'].values

    FEATURES = ['document_count', 'mean_document_length', 'document_length', 'term_frequency_1', 'term_frequency_2',
                'term_frequency_3', 'term_frequency_4', 'term_frequency_5', 'document_frequency_1',
                'document_frequency_2',
                'document_frequency_3', 'document_frequency_4', 'document_frequency_5']

    feature_cols = data_set[FEATURES].to_numpy()

    return feature_cols, labels

# read training data
filename = 'query.titles.csv'
training_input = './' + filename
df = pd.read_csv(training_input)

# get features and labels
feature_cols, labels = input_fn(df)

#need to change to real X_train, X_test!!
sp_line = int(len(feature_cols)*0.8)

X_train, y_train = feature_cols[:sp_line], labels[:sp_line]
X_test, y_test = feature_cols[sp_line:], labels[sp_line:]

# parameters
input_size = len(feature_cols)
batch_size = 512
hidden_units = 1024
dropout = 0.5
learning_rate = 1e-3

n_features = feature_cols.shape[1]

model = Sequential()
model.add(Dense(hidden_units, activation='relu', input_shape=(n_features,)))
model.add(Dropout(dropout))
model.add(Dense(hidden_units, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(1))

opt = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=opt, loss='mse')

model.fit(X_train, y_train, epochs=5, batch_size=batch_size, verbose=0)

rst = model.predict(X_test)
print(rst)
with open("model_output", "w") as f:
    for i in rst:
        for v in i:
            f.write(str(v))
            f.write('\n')
