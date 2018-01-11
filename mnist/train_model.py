#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/12 1:56
# @Author  : Wenbo Tang
# @File    : train_model.py

from load_data import Data
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.optimizers import SGD
from keras.utils import to_categorical

# 自动分配显存
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)


# load data
data = Data()
train_data, train_label = data.load_train_data()
train_data = train_data.reshape(-1, 28, 28, 1)
print("trainset:", train_data.shape, train_label.shape)

# test_data = data.load_test_data()
# test_data = test_data.reshape(-1, 1, 28, 28)
# print("testset:", test_data.shape)

# print(train_label[:10])
# print(test_label[:10])


# construct model

x_train = train_data
x_test = train_data[40000:]
y_train = train_label
y_test = train_label[40000:]

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# create model
model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(5, 5), padding='valid', input_shape=(28, 28, 1), activation='tanh'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='tanh'))
model.add(MaxPool2D(pool_size=(2, 2)))

# 池化后变成16个4x4的矩阵，然后把矩阵压平变成一维的，一共256个单元。
model.add(Flatten())
# 下面就是全连接层了
model.add(Dense(300, activation='tanh'))
model.add(Dense(100, activation='tanh'))
model.add(Dense(10, activation='softmax'))
# compile model

# 事实证明，对于分类问题，使用交叉熵(cross entropy)作为损失函数更好些
model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(lr=0.1),
    metrics=['accuracy']
)

# train model
model.fit(x_train, y_train, batch_size=128, epochs=40, verbose=2, shuffle=True)

# evaluate model

score = model.evaluate(x_test, y_test)
print("Total loss on Testing Set:", score[0])
print("Accuracy of Testing Set:", score[1])

model.save('./mnist_lenet.h5')
