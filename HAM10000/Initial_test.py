# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 02:41:24 2019

@author: darp_lord
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

rgb_28=pd.read_csv("../Datasets/hmnist_28_28_RGB.csv",delimiter=",")
train_X, test_X, train_Y, test_Y=train_test_split(rgb_28.iloc[:,:-1],rgb_28.iloc[:,-1],test_size=0.2,random_state=69)

#print(np.unique(train_Y), np.unique(test_Y))
num_rows=28
num_cols=28
batch_size = 64
nb_epoch = 10
num_classes = 7

train_X=train_X.values.reshape(train_X.shape[0],num_rows,num_cols,3)
train_Y=to_categorical(train_Y, num_classes)
model=Sequential([Conv2D(32,(3,3),activation="relu",input_shape=(num_rows,num_cols,3)),
				   MaxPool2D((2,2)),
				   Conv2D(64,(3,3),activation="relu"),
				   MaxPool2D((2,2)),
				   Flatten(),
				   Dense(128, activation="relu"),
				   Dropout(0.25),
				   Dense(num_classes,activation="softmax")])
model.compile(optimizer="adam", loss="binary_crossentropy",metrics=['accuracy','categorical_accuracy'])
model.fit(train_X,train_Y,epochs=nb_epoch,batch_size=batch_size)
model.save("first.h5")

test_X=test_X.values.reshape(test_X.shape[0],num_rows,num_cols,3)
test_Y=to_categorical(test_Y, num_classes)
print(model.evaluate(test_X,test_Y,batch_size=batch_size))
