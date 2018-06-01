#!/usr/bin/env python
# -*- coding: utf-8 -*-
from gensim.models import word2vec
import logging

# 主程序
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus(r"C:\Users\CJ17\Desktop\text8")
model = word2vec.Word2Vec(sentences, sg=1, size=200, window=10, min_count=5,
                     negative = 10)
model.save(r"C:\Users\CJ17\Desktop\text8output\200dimension\text8.model")
model.wv.save_word2vec_format(r"C:\Users\CJ17\Desktop\text8output\200dimension\text8.model.vector")

y1 = model.similarity("woman", "man")
print(u"woman and man :", y1)
'''
model.save("text8.model")# add the path into 
# 对应的加载方式
# model_2 = word2vec.Word2Vec.load("text8.model")

model.save_word2vec_format("text8.model.bin", binary=True)
# 对应的加载方式
# model_3 = word2vec.Word2Vec.load_word2vec_format("text8.model.bin", binary=True)
'''