# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 20:49:17 2019

@author: darp_lord
"""
import numpy as np
import gzip
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.utils import to_categorical

DATA_DIR="../Datasets/"
IMG_SIZE=28

images=gzip.open(DATA_DIR+"train-images-idx3-ubyte.gz","rb")
labels=gzip.open(DATA_DIR+"train-labels-idx1-ubyte.gz","rb")

def get_mnist_data(img_zip,lbl_zip):
	images=gzip.open(DATA_DIR+img_zip,"rb")
	labels=gzip.open(DATA_DIR+lbl_zip,"rb")

	magic_img=int.from_bytes(images.read(4),"big")
	magic_lbl=int.from_bytes(labels.read(4),"big")

	if magic_img!=2051 or magic_lbl!=2049:
		print("Error! Magic number does not match")
		return

	no_of_images=int.from_bytes(images.read(4),"big")
	no_of_labels=int.from_bytes(labels.read(4),"big")

	if no_of_images!=no_of_labels:
		print("Error! number of iamges and labels do not match")
		return
	img_size_rows=int.from_bytes(images.read(4),"big")
	img_size_cols=int.from_bytes(images.read(4),"big")

	img_arr=np.empty((no_of_images,img_size_rows,img_size_cols,1))
	lbl_arr=np.empty((no_of_images,),int)

	for i in range(0,no_of_images):
		img_arr[i]=np.frombuffer(images.read(img_size_rows*img_size_cols),dtype=np.uint8).reshape(img_size_rows,img_size_cols,1)
		lbl_arr[i]=np.frombuffer(labels.read(1),dtype=np.uint8)
		#img=Image.fromarray(img_arr[i].reshape(28,28)).convert("L")
		#img.save("test"+str(i)+".jpg")
	#print(img_arr.shape,lbl_arr.shape)
	#print(lbl_arr)
	images.close()
	labels.close()
	return img_arr,lbl_arr,img_size_rows,img_size_cols
#plt.subplot(28,28,1)
#plt.imshow(X.reshape(28,28),cmap=cm.Greys_r)
#plt.axis('off')
#plt.show()
if __name__=="__main__":
	batch_size = 64
	nb_classes = 10
	nb_epoch = 10
	num_classes = 10
	train_images, train_labels, train_rows, train_cols=get_mnist_data("train-images-idx3-ubyte.gz",
																      "train-labels-idx1-ubyte.gz")
	train_labels=to_categorical(train_labels,num_classes)
	model=Sequential([Conv2D(32,(3,3),activation="relu",input_shape=(train_rows,train_cols,1)),
				   MaxPool2D((2,2)),
				   Conv2D(64,(3,3),activation="relu"),
				   MaxPool2D((2,2)),
				   Flatten(),
				   Dense(128, activation="relu"),
				   Dropout(0.25),
				   Dense(num_classes,activation="softmax")])
	model.compile(optimizer="adam", loss="binary_crossentropy",metrics=['accuracy'])
	model.fit(train_images,train_labels,epochs=nb_epoch,batch_size=batch_size)
	model.save("first.h5")
	test_images, test_labels,_,_=get_mnist_data("t10k-images-idx3-ubyte.gz",
										 "t10k-labels-idx1-ubyte.gz")
	test_labels=to_categorical(test_labels,num_classes)
	print(model.evaluate(test_images,test_labels,batch_size=batch_size))