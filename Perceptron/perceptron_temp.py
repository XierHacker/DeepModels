import pandas as pd
import numpy as np
import tensorflow as tf

#load data
train_frame=pd.read_csv("../TestData/MNIST/train.csv")
#test_frame=pd.read_csv("../TestData/MNIST/test.csv")
#pop the labels and one-hot coding
train_labels_frame=train_frame.pop("label")
train_labels_frame=pd.get_dummies(data=train_labels_frame)



#print(train_frame.shape,train_labels_frame.shape)
X=train_frame.values
y=train_labels_frame.values


#graph
graph=tf.Graph()
with graph.as_default():
    weights=tf.Variable(initial_value=tf.zeros(shape=(784,10)),name="weigths")
    biases=tf.Variable(initial_value=tf.zeros(shape=(10,)),name="biases")
    X_p=tf.placeholder(dtype=tf.float32,shape=(None,784),name="X_p")
    y_p=tf.placeholder(dtype=tf.float32,shape=(None,10),name="y_p")
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_p,logits=tf.matmul(X_p,weights)+biases))
   # cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_p*tf.log(prob),reduction_indices=[1]))
    optimizer=tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    init_op=tf.global_variables_initializer()

#session
with tf.Session(graph=graph) as sess:
    sess.run(init_op)
    for i in range(20):
        print("epochs:",i)
        _,l=sess.run(fetches=[optimizer,loss],
                        feed_dict={X_p:X[i*100:i*100+100],y_p:y[i*100:i*100+100]})
        print(l)
