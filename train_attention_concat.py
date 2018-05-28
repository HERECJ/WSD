from __future__ import print_function

import codecs
import multiprocessing
import os
import pickle

import numpy as np
from keras.layers import (Input,LSTM, Bidirectional, Dense, Embedding,
                          RepeatVector, TimeDistributed,concatenate)
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score,  f1_score

from keras import backend as K
from attentionlayer import Attention
from concatlayer import ConcatCon
from keras.callbacks import EarlyStopping
from keras.optimizers import Adadelta


#one-layer-of-BiLSTM
#200D embedding matrix, and embedding is trainable
#droput(0.5)+batch normalization


os.environ['CUDA_VISIBLE_DEVICES'] = "3"

HIDDEN_SIZE = 256
BATCH_SIZE = 32
MAX_LEN = 65 # 测试集里下标最大的为64，采取这个作为序列最长,但是是从0开始计算的索引
EPOCHS = 40


VOCAB_FILE = r"E:\python_cj\datasets\vocabulary.txt"
LABEL_VOCAB_FILE = r"E:\python_cj\datasets\vocabulary_sensetag_full.txt"
INPUT_FILE = r"E:\python_cj\datasets\train3_input_x.txt"
TARGET_FILE = r"E:\python_cj\datasets\train3_output_y.txt"
TEST_INPUT_FILE = r"E:\python_cj\datasets\test_input_x.txt"
TEST_OUTPUT_FILE = r"E:\python_cj\datasets\test_output_y.txt"
INDEX_FILE = r"E:\python_cj\datasets\index.txt"
filename = r"E:\python_cj\datasets\text8.model.vector"

'''
VOCAB_FILE = "vocabulary.txt"
LABEL_VOCAB_FILE = "vocabulary_sensetag_full.txt"
INPUT_FILE = "train3_input_x.txt"
TARGET_FILE = "train3_output_y.txt"
TEST_INPUT_FILE = "test_input_x.txt"
TEST_OUTPUT_FILE = "test_output_y.txt"
INDEX_FILE = "index.txt"
filename = "text8.model.vector"
'''
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


earlystop = EarlyStopping(monitor='loss',patience=1)
opt = Adadelta(lr=1.0)


_input = Input(shape=(MAX_LEN,))

embed_input = embedding_layers(_input)
embed_input = BatchNormalization()(embed_input)

output = Bidirectional(LSTM(units=HIDDEN_SIZE,return_sequences=True))(embed_input)

attention = Attention()(output)

attention = BatchNormalization()(attention)
output = BatchNormalization()(output)

softmax_input = ConcatCon()([output,attention])
outs = TimeDistributed(Dense(units=LABEL_SIZE,activation='softmax'))(softmax_input)

model = Model(_input,outs)

model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
true_sequences = to_categorical(true_sequences, num_classes=LABEL_SIZE)
test_data = (predict_sequences,true_sequences)
steps_per = int(SAMPLE_SIZE / BATCH_SIZE) + 1

model.fit_generator(data_generator(X_train, Y_trian, batch_size=BATCH_SIZE),
                steps_per_epoch=steps_per, epochs=EPOCHS,callbacks=[metrics], validation_data=test_data, workers=4)

model.summary()
