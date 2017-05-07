import numpy as np
import tensorflow as tf
import pandas as pd
from perceptron import Perceptron


train_frame=pd.read_csv("../TestData/MNIST/train.csv")
test_frame=pd.read_csv("../TestData/MNIST/test.csv")
#pop the labels and one-hot coding
train_labels_frame=train_frame.pop("label")
train_labels_frame=pd.get_dummies(data=train_labels_frame)

#load model
percept=Perceptron()
percept.fit(X=train_frame.values,y=train_labels_frame.values)
