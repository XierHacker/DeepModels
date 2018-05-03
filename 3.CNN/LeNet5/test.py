import numpy as np
import tensorflow as tf
import pandas as pd
from lenet5 import LeNet5


train_frame=pd.read_csv("../../data/MNIST/train.csv")
test_frame=pd.read_csv("../../data/MNIST/test.csv")

#pop the labels and one-hot coding
train_labels_frame=train_frame.pop("label")
#trans format
train_frame=train_frame.astype(np.float32)
test_frame=test_frame.astype(np.float32)

trainSet=train_frame.values
testSet=test_frame.values
trainSet=np.reshape(a=trainSet,newshape=(-1,28,28,1))
testSet=np.reshape(a=testSet,newshape=(-1,28,28,1))
y=train_labels_frame.values

#load model
lenet=LeNet5()
lenet.fit(X=trainSet,y=y)

#predict
result=lenet.predict(X=test_frame.values)
print(result)