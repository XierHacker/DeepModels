import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#visualize
def showPic(featureMap,i):
    channels=featureMap.shape[3]
    #for channel in range(channels):
    plt.imshow(featureMap[i][:,:,2])
    plt.show()

#load data
train_frame=pd.read_csv("../TestData/MNIST/train.csv")
#test_frame=pd.read_csv("../TestData/MNIST/test.csv")
#pop the labels and one-hot coding
train_labels_frame=train_frame.pop("label")

#print(train_frame.shape,train_labels_frame.shape)
X=train_frame.values
X=np.reshape(a=X,newshape=(-1,28,28,1))
y=train_labels_frame.values

print(X.shape)


#graph
graph=tf.Graph()
with graph.as_default():
    X_p = tf.placeholder(dtype=tf.float32, shape=(None, 28,28,1), name="X_p")
    y_p = tf.placeholder(dtype=tf.float32, shape=(None, 10), name="y_p")


    #------------------------conv layer 1------------------------------------------

    filter_conv1=tf.Variable(initial_value=tf.truncated_normal(shape=(5,5,1,32)),
                            dtype=tf.float32,name="fiter_conv1")
    biases_conv1=tf.Variable(initial_value=tf.truncated_normal(shape=(32,)),
                       dtype=tf.float32,name="biases_conv1")
    #we get nx28x28x32 feature map
    featureMap_conv1=tf.nn.conv2d(input=X_p,filter=filter_conv1,
                             strides=[1,1,1,1],padding="SAME",name="conv1")+biases_conv1
    #pooing --> (nx14x14x32)
    pool_featureMap_conv1=tf.nn.max_pool(value=featureMap_conv1,ksize=[1,2,2,1],
                              strides=[1,2,2,1],padding="SAME",name="pooing_conv1")
    #activation as next layer's input nx14x14x32
    activation_conv1=tf.nn.relu(features=pool_featureMap_conv1,name="relu_conv1")

    #------------------------------------------------------------------------------



    # ------------------------conv layer 2------------------------------------------

    filter_conv2 = tf.Variable(initial_value=tf.truncated_normal(shape=(5, 5, 32, 64)),
                               dtype=tf.float32, name="fiter_conv2")
    biases_conv2 = tf.Variable(initial_value=tf.truncated_normal(shape=(64,)),
                               dtype=tf.float32, name="biases_conv2")
    # we get nx14x14x64 feature map
    featureMap_conv2 = tf.nn.conv2d(input=activation_conv1, filter=filter_conv2,
                                    strides=[1, 1, 1, 1], padding="SAME", name="conv2")+biases_conv2
    # pooing -->nx7x7x64
    pool_featureMap_conv2 = tf.nn.max_pool(value=featureMap_conv2, ksize=[1, 2, 2, 1],
                                           strides=[1, 2, 2, 1], padding="SAME", name="pooing_conv2")

    # activation as next layer's input nx7x7x64
    activation_conv2 = tf.nn.relu(features=pool_featureMap_conv2, name="relu_conv2")
    #reshape,because it will connect to fc layer
    activation_conv2=tf.reshape(tensor=activation_conv2,shape=(-1,7*7*64))
    # ------------------------------------------------------------------------------


    # ------------------------fully connected layer 1------------------------------------------

    weights_fc1 = tf.Variable(initial_value=tf.truncated_normal(shape=(7*7*64,1024)),
                              name="weights_fc1")
    biases_fc1 = tf.Variable(initial_value=tf.zeros(shape=(1024,)), name="biases_fc1")
    #we get (nx1024)
    activation_fc1=tf.nn.relu(features=tf.matmul(activation_conv2,weights_fc1)+biases_fc1)

    # ------------------------------------------------------------------------------


    # ------------------------fully connected layer 2(output layer)------------------------------------------

    weights_fc2 = tf.Variable(initial_value=tf.truncated_normal(shape=(1024, 10)),
                              name="weights_fc2")
    biases_fc2 = tf.Variable(initial_value=tf.zeros(shape=(10,)), name="biases_fc2")
    # we get (nx10)
    logits=tf.matmul(activation_fc1, weights_fc2) + biases_fc2

    # ------------------------------------------------------------------------------


    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_p,logits=logits,name="loss"))
    optimizer=tf.train.AdamOptimizer(0.1).minimize(loss)

    init_op = tf.global_variables_initializer()

#session
with tf.Session(graph=graph) as sess:
    y_dummy = pd.get_dummies(data=y).values
    sess.run(init_op)
    for i in range(40):
        print("epochs:", i)
        _, l = sess.run(fetches=[optimizer, loss],
                        feed_dict={X_p: X[i * 100:i * 100 + 100], y_p: y_dummy[i * 100:i * 100 + 100]})
        print(l)


