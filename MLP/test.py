import numpy as np
import tensorflow as tf
import pandas as pd
from mlp import MLP

'''

'''




train_frame=pd.read_csv("../TestData/MNIST/train.csv")
test_frame=pd.read_csv("../TestData/MNIST/test.csv")

#pop the labels and one-hot coding
train_labels_frame=train_frame.pop("label")
#trans format
train_frame=train_frame.astype(np.float32)
test_frame=test_frame.astype(np.float32)

#load model
model=MLP(300,100)
model.fit(X=train_frame.values,y=train_labels_frame.values)

#predict
#result=percept.predict(X=test_frame.values)
#print(result)