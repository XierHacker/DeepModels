import numpy as np
import tensorflow as tf
import pandas as pd
from perceptron import Perceptron

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
percept=Perceptron()
percept.fit(X=train_frame.values,y=train_labels_frame.values)

#predict
result=percept.predict(X=test_frame.values)
print(result)


'''
g=tf.Graph()
with g.as_default():
    t1 = tf.constant([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 3, 5, 4, 2], [2, 2, 2, 4, 1], [7, 7, 7, 8, 2]])
    arg=tf.argmax(input=t1,axis=1)
#print(5//2)
#print(list(range(0,3)))


with tf.Session(graph=g) as sess:
    result=sess.run(fetches=arg)
    print(result)
    
a=np.array([1,2,3,4,4,1])
a_dummy=pd.get_dummies(a)
print(a_dummy)
'''

