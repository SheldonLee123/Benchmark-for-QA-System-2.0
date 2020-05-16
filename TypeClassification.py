import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.layers import BatchNormalization
from keras.layers.merge import concatenate

df = pd.read_csv('./dataset/Type.csv')
# print(df.head())
# print(df.Type.value_counts())

title = df.Question
label = df.Type
X_train, X_test, y_train, y_test = train_test_split(title, label, test_size=0.1, random_state=42)

# MultinomialNB Classifier

# vect = TfidfVectorizer(stop_words='english', token_pattern=r'\b\w{2,}\b', min_df=1, max_df=0.1, ngram_range=(1,2))                          # r: Raw String 字符串不会转义
# mnb = MultinomialNB(alpha=2)              # alpha 平滑参数
# svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42)         #random_state参数的作用是为了保证每次运行程序时都以同样的方式进行分割
# mnb_pipeline = make_pipeline(vect, mnb)
# svm_pipeline = make_pipeline(vect, svm)
# mnb_cv = cross_val_score(mnb_pipeline, title, label, scoring='accuracy', cv=10, n_jobs=-1)
# svm_cv = cross_val_score(svm_pipeline, title, label, scoring='accuracy', cv=10, n_jobs=-1)
# print('\nMultinomialNB Classifier\'s Accuracy: %0.5f\n' % mnb_cv.mean())
# print('\nSVM Classificer\'s Accuracy: %0.5f\n' % svm_cv.mean())


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
x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=20)                #将序列转化为经过填充以后的一个长度相同的新序列
x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=20)

def DenseModel():
    # one-hot mlp
    x_train = tokenizer.sequences_to_matrix(x_train_word_ids, mode='binary')
    x_test = tokenizer.sequences_to_matrix(x_test_word_ids, mode='binary')

    model = Sequential()
    model.add(Dense(512, input_shape=(len(vocab)+1,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                                     optimizer='adam',
                                     metrics=['accuracy'])

    model.fit(x_train, y_train,
                         batch_size=32,
                         epochs=15,
                         validation_data=(x_test, y_test))

    # X_predict = ["who was the american general in the pacific during world war ii","where do guyanese people live","what is magic johnsons dads name"]
    # model = load_model('./model/DenseModel.h5')
    # x_predict_word_ids = tokenizer.texts_to_sequences(X_predict)
    # x_predict = tokenizer.sequences_to_matrix(x_predict_word_ids, mode='binary')
    # predict_test = model.predict(x_predict)
    # print(np.argmax(predict_test,axis=1))

    loss, accuracy = model.evaluate(x_test, y_test)
    print('\ntest loss: ', loss)
    print('\ntest accuracy: ', accuracy)

    loss, accuracy = model.evaluate(x_train, y_train)
    print('\ntrain loss: ', loss)
    print('\ntrain accuracy: ', accuracy)

    model.save('./TemporalModel/DenseModel.h5')

# RNN model
def RNN():
    model = Sequential()
    model.add(Embedding(len(vocab)+1, 256, input_length=20))
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.1))
    model.add(Dense(num_labels, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                                     optimizer='adam',
                                     metrics=['accuracy'])

    model.fit(x_train_padded_seqs, y_train,
                         batch_size=32,
                         epochs=12,
                         validation_data=(x_test_padded_seqs, y_test))

    loss, accuracy = model.evaluate(x_test_padded_seqs, y_test)
    print('\ntest loss: ', loss)
    print('\ntest accuracy: ', accuracy)

    model.save('./TemporalModel/RNNModel.h5')

# CNN model
def CNN():
    model = Sequential()
    model.add(Embedding(len(vocab)+1, 256, input_length=20))

    # Convolutional moedl (3x conv, flatten, 2x dense)
    model.add(Convolution1D(256, 3, padding='same'))
    model.add(MaxPool1D(3, 3, padding='same'))
    model.add(Convolution1D(128, 3, padding='same'))
    model.add(MaxPool1D(3, 3, padding='same'))
    model.add(Convolution1D(64, 3, padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_labels, activation='softmax'))

    model.compile(loss = 'categorical_crossentropy',
                                     optimizer='adam',
                                     metrics=['accuracy'])

    model.fit(x_train_padded_seqs, y_train,
                         batch_size=32,
                         epochs=12,
                         validation_data=(x_test_padded_seqs, y_test))

    loss, accuracy = model.evaluate(x_test_padded_seqs, y_test)
    print('\ntest loss: ', loss)
    print('\ntest accuracy: ', accuracy)

    loss, accuracy = model.evaluate(x_train_padded_seqs, y_train)
    print('\ntrain loss: ', loss)
    print('\ntrain accuracy: ', accuracy)

    model.save('./TemporalModel/CNNModel.h5')

# TextCNN
def TextCNN():
    main_input = Input(shape=(20,), dtype='float64')
    embedder = Embedding(len(vocab) + 1, 300, input_length = 20)
    embed = embedder(main_input)
    cnn1 = Convolution1D(256, 3, padding='same', strides = 1, activation='relu')(embed)
    cnn1 = MaxPool1D(pool_size=4)(cnn1)
    cnn2 = Convolution1D(256, 4, padding='same', strides = 1, activation='relu')(embed)
    cnn2 = MaxPool1D(pool_size=4)(cnn2)
    cnn3 = Convolution1D(256, 5, padding='same', strides = 1, activation='relu')(embed)
    cnn3 = MaxPool1D(pool_size=4)(cnn3)
    cnn = concatenate([cnn1,cnn2,cnn3], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    main_output = Dense(num_labels, activation='softmax')(drop)
    model = Model(inputs = main_input, outputs = main_output)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train_padded_seqs, y_train,
              batch_size=32,
              epochs=12,
              validation_data=(x_test_padded_seqs, y_test))

    loss, accuracy = model.evaluate(x_test_padded_seqs, y_test)
    print('\ntest loss: ', loss)
    print('\ntest accuracy: ', accuracy)

    model.save('./TemporalModel/TextCNNModel.h5')

if __name__ == '__main__':
    # DenseModel()
    CNN()
    # RNN()
    # TextCNN()