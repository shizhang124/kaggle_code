#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/12 2:57
# @Author  : Wenbo Tang
# @File    : predict.py
from load_data import Data
from keras.models import load_model
import numpy as np


# 自动分配显存
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)


data = Data()
test_data = data.load_test_data()
test_data = test_data.reshape(-1, 28, 28, 1)

model = load_model("./mnist_lenet.h5")
results = model.predict_classes(test_data, batch_size=256, verbose=1)

results = results.reshape(-1, 1)
index = np.array(range(1, 28001)).reshape(-1, 1)

print(index.shape, results.shape)
result = np.concatenate((index, results), axis=1)
data.wirte_result(result)
