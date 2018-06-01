#!/usr/bin/env python
# -*- coding: utf-8 -*-
#https://blog.csdn.net/lxg0807/article/details/72518962
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from nltk.corpus import semcor
def loadWord2Vec(filename):
    vocab = []
    embd = []
    cnt = 0
    fr = open(filename,'r')
    line = fr.readline().strip()
    #print(line)
    word_dim = int(line.split(' ')[1])
    vocab.append("unk")
    embd.append([0]*word_dim)
    for line in fr:
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print("loaded word2vec")
    fr.close()
    return vocab,embd

filename = r"C:\Users\CJ17\Desktop\text8output\text8.model.vector"
vocab,embd = loadWord2Vec(filename)
vocab_size = len(vocab)
embedding_dim = len(embd[0])
embedding = np.asarray(embd)
print(embedding_dim)


#在网络中构建词向量单元
W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                trainable=False, name="W")
embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
embedding_init = W.assign(embedding_placeholder)

sess = tf.Session()
sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})

#用embedding_lookup函数将输入的单词映射为词向量，但是这个过程中需要单词的下标
#用api对原来的文本进行处理，但是这个过程会过滤标点符号

words = semcor.words('brown2/tagfiles/br-e30.xml')
data = []
for word in words:
    data.append(word)
print("finishing reading raw inputs")
input_of_raw = data
max_document_length=1000000
#init vocab processor
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length,min_frequency=3)
#fit the vocab from glove
pretrain = vocab_processor.fit(vocab)
#transform inputs
input_x = np.array(list(vocab_processor.transform(input_of_raw)))

tf.nn.embedding_lookup(W, input_x)

