# coding: utf-8
from tqdm import trange
import pandas as pd
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical

df = pd.read_csv('./dataset/PosNeg2.0.csv')
title = df.question
label = df.label
X_train, X_test, y_train, y_test = train_test_split(title, label, test_size=0.1, random_state=42)

y_labels = list(y_train.value_counts().index)
le = preprocessing.LabelEncoder()
le.fit(y_labels)
num_labels = len(y_labels)
y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), num_labels)
y_test = to_categorical(y_test.map(lambda x: le.transform([x])[0]), num_labels)

# load glove word embedding data
GLOVE_DIR = "./glove.6B"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), encoding = 'utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# take tokens and build word-in dictionary
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
tokenizer.fit_on_texts(title)
vocab = tokenizer.word_index

# Match the word vector for each word in the data set from Glove
embedding_matrix = np.zeros((len(vocab)+1, 300))
for word, i in vocab.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Match the input format of the model
x_train_word_ids = tokenizer.texts_to_sequences(X_train)               #序列的列表，列表中每个序列对应于一段输入文本
x_test_word_ids = tokenizer.texts_to_sequences(X_test)
x_train = tokenizer.sequences_to_matrix(x_train_word_ids, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test_word_ids, mode='binary')
x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=20)                #将序列转化为经过填充以后的一个长度相同的新序列
x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=20)

# model = load_model('./Model/CNNModel2.0.h5')
# loss, accuracy = model.evaluate(x_train_padded_seqs, y_train)
# print('\ntrain loss: ', loss)
# print('\ntrain accuracy: ', accuracy)
#
# loss, accuracy = model.evaluate(x_test_padded_seqs, y_test)
# print('\ntest loss: ', loss)
# print('\ntest accuracy: ', accuracy)

# Predict DenseModel
# X_predict = ["who was the american general in the pacific during world war ii","where do guyanese people live","what is magic johnsons dads name"]
model = load_model('./model/DenseModel2.0.h5')
loss, accuracy = model.evaluate(x_train, y_train)
print('\ntrain loss: ', loss)
print('\ntrain accuracy: ', accuracy)

loss, accuracy = model.evaluate(x_test, y_test)
print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

# x_predict_word_ids = tokenizer.texts_to_sequences(X_predict)
# x_predict = tokenizer.sequences_to_matrix(x_predict_word_ids, mode='binary')
# predict_test = model.predict(x_predict)
# predict_result = np.argmax(predict_test,axis=1)                           # 1 temporal   0 no-temporal
# print(predict_result)
# print(len(predict_result))
#
# #Predict CNN
#
# df = pd.read_csv('./dataset/lcquad2/LcQuadV2_question_with_id.csv')
# df.head()
#
# df['question'].isnull().sum()
#
# df.dropna(subset=['question'],inplace=True)
#
# X_predict = df.question
#
# # X_predict = ["who was the first female athlete to be on the wheaties box?", "where was first enclosed nfl stadium?", "when did reagan first run for president?", "what movie did julie andrews first star in?"]
# x_predict_word_ids = tokenizer.texts_to_sequences(X_predict)
# x_predict = pad_sequences(x_predict_word_ids, maxlen=20)
# predict_test = model.predict(x_predict)
# predict_result = np.argmax(predict_test,axis=1)                           # 0 Explicit 1 Implicit  2 Ordinal  3 Temp.Ans
# print(predict_result)
# print(len(predict_result))
