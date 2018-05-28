from __future__ import print_function

import codecs
import multiprocessing
import os
import pickle
import sys

import numpy as np
from keras.layers import (LSTM, Bidirectional, Dense, Dropout, Embedding,
                          Input, RepeatVector, TimeDistributed)
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.utils.vis_utils import plot_model
from sklearn.metrics import accuracy_score,  f1_score
from keras import backend as K
import tensorflow as tf

#one-layer-of-BiLSTM
#200D embedding matrix, and embedding is trainable
#droput(0.5)+batch normalization


os.environ['CUDA_VISIBLE_DEVICES'] = '5'

HIDDEN_SIZE = 256
BATCH_SIZE = 32
MAX_LEN = 65 # 测试集里下标最大的为64，采取这个作为序列最长,但是是从0开始计算的索引
EPOCHS = 40


VOCAB_FILE = "vocabulary.txt"
LABEL_VOCAB_FILE = "vocabulary_sensetag_full.txt"
INPUT_FILE = "train3_input_x.txt"
TARGET_FILE = "train3_output_y.txt"
TEST_INPUT_FILE = "test_input_x.txt"
TEST_OUTPUT_FILE = "test_output_y.txt"
INDEX_FILE = "index.txt"


# 数据生成以及迭代器的构造

# the embedding layer
def loadWord2Vec(filename):
    vocab = []
    embd = []
    cnt = 0
    fr = open(filename, 'r')
    line = fr.readline().strip()
    word_dim = int(line.split(' ')[1])
    vocab.append(0)
    embd.append([0] * word_dim)
    # print(line)
    word_dim = int(line.split(' ')[1])
    for line in fr:
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])

    print("loaded word2vec")
    vocab.append("unk")
    embd.append([0] * word_dim)
    fr.close()
    return vocab, embd


filename = "text8.model.vector"
vocab, embd = loadWord2Vec(filename)
vocab_size = len(vocab)
embedding_dim = len(embd[0])
embedding_matrix = np.asarray(embd)
print(embedding_dim)

embedding_layers = Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                             weights=[embedding_matrix], input_length=MAX_LEN, trainable=True, mask_zero=True)


def Sequence_Generate(VOCAB, RAW):
    texts = []
    with codecs.open(VOCAB, 'r', 'utf-8') as fn:
        vocab = [w.strip() for w in fn.readlines()]
    word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

    def get_id(word):
        return word_to_id[word] if word in word_to_id else word_to_id["unk"]

    with open(RAW, 'r') as fout:
        lines = fout.readlines()
        for line in lines:
            s = line.split()
            text = []
            for data in s:
                text.append(get_id(data))
            # print(output_text)
            texts.append(text)
    texts = pad_sequences(texts, maxlen=MAX_LEN, padding='post')

    return texts, vocab


input_sequences, _ = Sequence_Generate(VOCAB_FILE, INPUT_FILE)
output_sequences, vocab_labels = Sequence_Generate(LABEL_VOCAB_FILE, TARGET_FILE)
predict_sequences, _ = Sequence_Generate(VOCAB_FILE, TEST_INPUT_FILE)
true_sequences , _ = Sequence_Generate(LABEL_VOCAB_FILE,TEST_OUTPUT_FILE)

# VOCAB_SIZE = vocab_size
LABEL_SIZE = len(vocab_labels)
SAMPLE_SIZE = len(input_sequences)
# input_sequences 和 output_sequences 就是整个训练集（已经完成）
# 下面进行数据集的打乱

index = np.arange(SAMPLE_SIZE)
np.random.shuffle(index)
X_train = input_sequences[index]
Y_trian = output_sequences[index]



def data_generator(input_data, output_data, batch_size):
    while 1:
        steps = int(len(input_data) / batch_size)
        # print(steps)
        for i in range(steps):
            train_x = input_data[i * batch_size: (i + 1) * batch_size]
            train_y = output_data[i * batch_size: (i + 1) * batch_size]
            train_y = to_categorical(train_y, num_classes=LABEL_SIZE)
            yield (train_x, train_y)


fr = open(INDEX_FILE,'rb')
b = pickle.load(fr)
fr.close()
LINES = b[0]
NUMBERS = b[1]

#def My_Loss(y_true, y_pred,e =0.1):
#    return (1-e)*K.categorical_crossentropy(y_pred,y_true)+e*K.categorical_crossentropy(y_pred,K.ones_like(y_pred)/LABEL_SIZE)




class Metrics(Callback):
    def on_epoch_end(self, epoch, logs={}):
        predict = np.asarray(self.model.predict(self.validation_data[0]))
        #predict = np.argmax(predict,axis=2)
        predict = np.argmax(predict,axis = -1)
        targ = np.asarray(self.validation_data[1])
        targ = np.argmax(targ,axis=-1)

        true_data = []
        pred_data = []
        for i in range(len(LINES)):
            true_data.append(targ[LINES[i]][NUMBERS[i]])
            pred_data.append(predict[LINES[i]][NUMBERS[i]])

        #self.f1s=f1_score(targ, predict)
        self.f1s = f1_score(true_data,pred_data,average='micro')
        self.accu = accuracy_score(true_data,pred_data)
        print("f1_score: %.3f ,acc: %.3f" %(self.f1s,self.accu))
        return

metrics = Metrics()


model = Sequential()
model.add(embedding_layers)
# 构建双层的双向的LSTM层作为解码器
#model.add(BatchNormalization())

model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True, dropout=0.5)))
model.add(BatchNormalization())

#model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True,dropout=0.5)))
#model.add(BatchNormalization())


model.add(TimeDistributed(Dense(LABEL_SIZE, activation='softmax')))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
#model.compile(optimizer='adam',loss=My_Loss,metrics=['accuracy'])

#plot_model(model, to_file='bilstm_1_layer.png',show_shapes=True)
true_sequences = to_categorical(true_sequences, num_classes=LABEL_SIZE)
test_data = (predict_sequences,true_sequences)
steps_per = int(SAMPLE_SIZE / BATCH_SIZE) + 1
model.fit_generator(data_generator(X_train, Y_trian, batch_size=BATCH_SIZE),
                steps_per_epoch=steps_per, epochs=EPOCHS,callbacks=[metrics], validation_data=test_data, workers=multiprocessing.cpu_count())

model.summary()
