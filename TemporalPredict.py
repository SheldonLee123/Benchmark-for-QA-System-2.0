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
from multiprocessing import Process
import csv

#Get the id of the question
def checkId(predict, df):
    for i in range(0, len(df['question'])):
        if predict == df['question'][i]:
            return df['uid'][i]

def WriteData(tempfile, nontempfile, start, end, X_predict, predict_result, df):
    # f0 = open(nontempfile, "w", encoding="utf-8")
    f1 = open(tempfile, "w", encoding="utf-8", newline='')

    # csv_writer0 = csv.writer(f0)
    csv_writer1 = csv.writer(f1)

    # csv_writer0.writerow(["uid", "question"])
    csv_writer1.writerow(["uid", "question"])

    for i in trange(start, end):            # try to use the processorbar
        # print(i)
        id = str(checkId(X_predict.iloc[i], df))
        # if predict_result[i] == 0:
        #     csv_writer0.writerow([id, X_predict.iloc[i]])
            # f0.write(X_predict.iloc[i] + "\n")
        if predict_result[i] == 1:
            csv_writer1.writerow([id, X_predict.iloc[i]])
            # f1.write(X_predict.iloc[i] + "\n")
    # f0.close()
    f1.close()

def combineAll(tempfile, nontempfile):
    with open("./dataset/lcquad2/output/id-temporal.csv", "w", encoding="utf-8", newline='') as alltemp_file:
        csv_writer = csv.writer(alltemp_file)
        csv_writer.writerow(['uid', 'question'])
        for i in tempfile:
            with open(i, "r", encoding="utf-8") as temp_file:
                csv_reader = csv.reader(temp_file)
                header = next(csv_reader)
                for row in csv_reader:
                    csv_writer.writerow(row)

    # with open("./dataset/lcquad2/output/id-no-temporal.txt", "w", encoding="utf-8") as allnontemp_file:
    #     for i in nontempfile:
    #         with open(i, "r", encoding="utf-8") as nontemp_file:
    #             while True:
    #                 lines = nontemp_file.readline()
    #                 if not lines:
    #                     break
    #                 allnontemp_file.write(lines)

if __name__ == '__main__':
    # df = pd.read_csv('./dataset/PosNeg2.0.csv')
    # title = df.question
    # label = df.label
    # X_train, X_test, y_train, y_test = train_test_split(title, label, test_size=0.1, random_state=42)
    #
    # y_labels = list(y_train.value_counts().index)
    # le = preprocessing.LabelEncoder()
    # le.fit(y_labels)
    # num_labels = len(y_labels)
    # y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), num_labels)
    # y_test = to_categorical(y_test.map(lambda x: le.transform([x])[0]), num_labels)
    #
    # # load glove word embedding data
    # GLOVE_DIR = "./glove.6B"
    # embeddings_index = {}
    # f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), encoding='utf-8')
    # for line in f:
    #     values = line.split()
    #     word = values[0]
    #     coefs = np.asarray(values[1:], dtype='float32')
    #     embeddings_index[word] = coefs
    # f.close()
    #
    # # take tokens and build word-in dictionary
    # tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
    # tokenizer.fit_on_texts(title)
    # vocab = tokenizer.word_index
    #
    # # Match the word vector for each word in the data set from Glove
    # embedding_matrix = np.zeros((len(vocab) + 1, 300))
    # for word, i in vocab.items():
    #     embedding_vector = embeddings_index.get(word)
    #     if embedding_vector is not None:
    #         embedding_matrix[i] = embedding_vector
    #
    # # Match the input format of the model
    # x_train_word_ids = tokenizer.texts_to_sequences(X_train)  # 序列的列表，列表中每个序列对应于一段输入文本
    # x_test_word_ids = tokenizer.texts_to_sequences(X_test)
    # x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=20)  # 将序列转化为经过填充以后的一个长度相同的新序列
    # x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=20)
    #
    # model = load_model('./Model/CNNModel2.0.h5')
    # loss, accuracy = model.evaluate(x_train_padded_seqs, y_train)
    # print('\ntrain loss: ', loss)
    # print('\ntrain accuracy: ', accuracy)
    #
    # # Predict DenseModel
    # # X_predict = ["who was the american general in the pacific during world war ii","where do guyanese people live","what is magic johnsons dads name"]
    # # model = load_model('./model/DenseModel.h5')
    # # x_predict_word_ids = tokenizer.texts_to_sequences(X_predict)
    # # x_predict = tokenizer.sequences_to_matrix(x_predict_word_ids, mode='binary')
    # # predict_test = model.predict(x_predict)
    # # predict_result = np.argmax(predict_test,axis=1)                           # 1 temporal   0 no-temporal
    # # print(predict_result)
    # # print(len(predict_result))
    #
    # # Predict CNN
    #
    # df = pd.read_csv('./dataset/lcquad2/lcquad_2_0_id.csv')
    # df.head()
    #
    # df['question'].isnull().sum()
    #
    # df.dropna(subset=['question'], inplace=True)
    #
    # X_predict = df.question
    #
    # # X_predict = ["Is Juan José Ibarretxe a chairperson of FC Barcelona?", "Is Alexander Hamilton a lawyer?"]
    # x_predict_word_ids = tokenizer.texts_to_sequences(X_predict)
    # x_predict = pad_sequences(x_predict_word_ids, maxlen=20)
    # predict_test = model.predict(x_predict)
    # # predict_pro = model.predict_proba(x_predict)
    # # print(predict_pro)
    # predict_result = np.argmax(predict_test, axis=1)  # 0 nontemporal 1 temporal
    # print(predict_result)
    # print(len(predict_result))
    #
    tempfile = []
    nontempfile = []
    # start = []
    # end = []
    for i in range(8):
        tempfile.append("./dataset/lcquad2/output/id-temporal" + str(i) +".csv")
        nontempfile.append("./dataset/lcquad2/output/id-no-temporal" + str(i) +".csv")
    #     start.append(int(30076/8*i))
    #     end.append(int(30076/8*(i+1)))
    #
    # for i in range(8):
    #     p = Process(target=WriteData ,args=(tempfile[i], nontempfile[i], start[i], end[i], X_predict, predict_result, df))
    #     p.start()

    combineAll(tempfile, nontempfile)