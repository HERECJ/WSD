from __future__ import print_function

import codecs
import multiprocessing
import os
import pickle
import re


import numpy as np
from keras.layers import (LSTM, Bidirectional, Dense, Lambda, Embedding,
                          Input, RepeatVector, TimeDistributed,concatenate)
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.utils.vis_utils import plot_model
from sklearn.metrics import accuracy_score,  f1_score
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adadelta
from losslayer import MyLossCom


#one-layer-of-BiLSTM
#200D embedding matrix, and embedding is trainable
#droput(0.5)+batch normalization


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

HIDDEN_SIZE = 256
BATCH_SIZE = 32
MAX_LEN = 65 # 测试集里下标最大的为64，采取这个作为序列最长,但是是从0开始计算的索引
EPOCHS = 40


VOCAB_FILE = r"E:\python_cj\datasets\vocabulary.txt"
LABEL_VOCAB_FILE = r"E:\python_cj\datasets\/vocabulary_sensetag_full.txt"
INPUT_FILE = r"E:\python_cj\datasets\train3_input_x.txt"
TARGET_FILE = r"E:\python_cj\datasets\train3_output_y.txt"
TEST_INPUT_FILE = r"E:\python_cj\datasets\test_input_x.txt"
TEST_OUTPUT_FILE = r"E:\python_cj\datasets\test_output_y.txt"
INDEX_FILE = r"E:\python_cj\datasets\index.txt"
filename = r"E:\python_cj\datasets\text8.model.vector"

# 数据生成以及迭代器的构造

# the embedding layer
def loadWord2Vec(filename):
    vocab = []
    embd = []
    #cnt = 0
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


def Sequence_Generate(VOCAB, RAW,is_out=False):
    texts = []

    with codecs.open(VOCAB, 'r', 'utf-8') as fn:
        vocab = [w.strip() for w in fn.readlines()]

    word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

    def get_id(word):
        return word_to_id[word] if word in word_to_id else word_to_id["unk"]

    if is_out:
        index_senses = []
        with open(RAW, 'r') as fout:
            lines = fout.readlines()
            for line in lines:
                s = line.split()
                text = []
                index_sense = []
                for data in s:
                    text.append(get_id(data))
                    if re.match(r'^(\d|[a-z])+(\w)*(\-)*(\d[a-z]|[a-z]\d)*([a-z])+([0-9]{2})\b',data):
                        index_sense.append(1)
                    else:
                        index_sense.append(0.5) #donnot use bool tensor(impact the padding,so zero is unavailable)
                texts.append(text)
                index_senses.append(index_sense)
        index_senses = pad_sequences(index_senses,maxlen=MAX_LEN,padding='post')

    else:
        with open(RAW, 'r') as fout:
            lines = fout.readlines()
            for line in lines:
                s = line.split()
                text = []
                #index_sense = []
                for data in s:
                    text.append(get_id(data))
                # print(output_text)
                texts.append(text)
        index_senses = []

    texts = pad_sequences(texts, maxlen=MAX_LEN, padding='post')


    return texts, vocab,index_senses


input_sequences, _ , _= Sequence_Generate(VOCAB_FILE, INPUT_FILE)
output_sequences, vocab_labels,index_sequences = Sequence_Generate(LABEL_VOCAB_FILE, TARGET_FILE,is_out=True)
test_input, _ , _ = Sequence_Generate(VOCAB_FILE, TEST_INPUT_FILE)
test_output , _ , test_index = Sequence_Generate(LABEL_VOCAB_FILE,TEST_OUTPUT_FILE,is_out=True)

# VOCAB_SIZE = vocab_size
LABEL_SIZE = len(vocab_labels)
SAMPLE_SIZE = len(input_sequences)

# input_sequences 和 output_sequences 就是整个训练集（已经完成）
# 下面进行数据集的打乱

index = np.arange(SAMPLE_SIZE)
np.random.shuffle(index)
X_train = input_sequences[index]
Y_trian = output_sequences[index]
Index_Train = index_sequences[index]


def data_generator(input_data, output_data, index_data, batch_size):
    while 1:
        steps = int(len(input_data) / batch_size)
        # print(steps)
        for i in range(steps):
            train_x = input_data[i * batch_size: (i + 1) * batch_size]
            train_y = output_data[i * batch_size: (i + 1) * batch_size]
            train_index = index_data[i * batch_size: (i + 1) * batch_size]
            train_index = np.expand_dims(train_index,-1)
            train_y = to_categorical(train_y, num_classes=LABEL_SIZE)
            train_y = np.concatenate((train_index,train_y),axis=-1)
            yield ([train_x, train_index],train_y)


fr = open(INDEX_FILE,'rb')
b = pickle.load(fr)
fr.close()
LINES = b[0]
NUMBERS = b[1]

def My_Loss(y_true, y_pred):
    index_true = y_true[...,0]
    target=y_true[...,1:]
    #index_pred = y_pred[...,0]
    output = y_pred[...,1:]



    #seed = K.random_uniform_variable(shape=(MAX_LEN,),low=0,high=2,dtype='int32')
    #seed = K.cast(seed,dtype='float32')
    #weight_value = index_true * seed

    weight_value = index_true
    tensor_value = K.expand_dims(weight_value,-1)
    tensor_value = K.tile(tensor_value,[1,1,LABEL_SIZE])


    target = tensor_value * target
    return K.categorical_crossentropy(target,output)

def acc_pred(y_true,y_pred):
    target = y_true[...,1:]
    output = y_pred[...,1:]
    return K.cast(K.equal(K.argmax(target,-1),
                          K.argmax(output,-1)),
                  K.floatx())

class Metrics(Callback):
    def on_epoch_end(self, epoch, logs={}):
        predict = np.asarray(self.model.predict([self.validation_data[0],self.validation_data[1]]))
        predict = predict[...,1:]
        predict = np.argmax(predict,axis = -1)
        targ = np.asarray(self.validation_data[2])
        #index_labels = targ[...,0:1]
        targ = targ[...,1:]
        targ = np.argmax(targ,axis=-1)

        true_data = []
        pred_data = []
        for i in range(len(LINES)):
            true_data.append(targ[LINES[i]][NUMBERS[i]])
            pred_data.append(predict[LINES[i]][NUMBERS[i]])

        self.f1s = f1_score(true_data,pred_data,average='micro')
        self.accu = accuracy_score(true_data,pred_data)
        print("f1_score: %.3f ,acc: %.3f" %(self.f1s,self.accu))
        return



metrics = Metrics()



earlystop = EarlyStopping(monitor='loss',patience=1)
opt = Adadelta(lr=1.0)

_input = Input(shape=(MAX_LEN,))
embed = embedding_layers(_input)
embed = BatchNormalization()(embed)

lstm_out = Bidirectional(LSTM(HIDDEN_SIZE,return_sequences=True,dropout=0.5))(embed)
lstm_out = BatchNormalization()(lstm_out)


output_labels = TimeDistributed(Dense(LABEL_SIZE,activation='softmax'))(lstm_out)

_input_index= Input(shape=(MAX_LEN,1),name='index_tensor')


outputs = concatenate([_input_index,output_labels],axis=-1)
model = Model(inputs=[_input,_input_index],outputs=outputs)

#model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])
model.compile(optimizer=opt,loss=My_Loss, metrics=['accuracy',acc_pred])
model.summary()
#plot_model(model, to_file='bilstm_1_layer.png',show_shapes=True)
test_output = to_categorical(test_output, num_classes=LABEL_SIZE)
test_index = np.expand_dims(test_index,axis=-1)
test_output = np.concatenate((test_index,test_output),axis=-1)
test_data = ([test_input,test_index],test_output)
steps_per = int(SAMPLE_SIZE / BATCH_SIZE) + 1

#model.fit_generator(data_generator(input_data=X_train, output_data=Y_trian,index_data=Index_Train, batch_size=BATCH_SIZE),
#                steps_per_epoch=steps_per, epochs=EPOCHS,callbacks=[metrics],
#                validation_data=test_data, workers=3)

model.fit_generator(data_generator(input_data=X_train, output_data=Y_trian,index_data=Index_Train, batch_size=BATCH_SIZE),
                steps_per_epoch=steps_per, epochs=EPOCHS, validation_data=test_data,callbacks=[metrics],workers=8)


model.summary()
