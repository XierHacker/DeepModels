import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
import perceptron
from utility import preprocessing


MAX_EPOCH=10
BATCH_SIZE=20
LEARNING_RATE=0.001
MODEL_SAVING_PATH="./saved_models/model.ckpt"

#load data
X_train,y_train,X_valid,y_valid,X_test=preprocessing.load_mnist(path="../../data/mnist/")
train_samples=X_train.shape[0]
valid_samples=X_valid.shape[0]

def train():
    #data placeholder
    X_p=tf.placeholder(dtype=tf.float32,shape=(None,perceptron.INPUT_DIM),name="X_p")
    y_p=tf.placeholder(dtype=tf.int32,shape=(None,),name="y_p")
    y_hot_p=tf.one_hot(indices=y_p,depth=perceptron.OUTPUT_DIM)

    #use regularizer
    regularizer=tf.contrib.layers.l2_regularizer(0.0001)

    #model
    model=perceptron.Perceptron()
    logits=model.forward(X_p,regularizer)           #[batch_size,10]
    pred=tf.argmax(input=logits,axis=-1)            #[batch_size,]

    #collect_list=tf.get_collection(key="regularized")
    #print("collect_list.shape",collect_list)
    loss=tf.losses.softmax_cross_entropy(onehot_labels=y_hot_p,logits=logits)+\
                        tf.add_n(inputs=tf.get_collection(key="regularized"))
    optimizer=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss=loss)

    init=tf.global_variables_initializer()

    #Saver class
    saver=tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(MAX_EPOCH):
            ls = []
            accus=[]
            for j in range(train_samples // BATCH_SIZE):
                _, l ,prediction= sess.run(
                    fetches=[optimizer, loss,pred],
                    feed_dict={
                        X_p: X_train[j * BATCH_SIZE:(j + 1) * BATCH_SIZE],
                        y_p: y_train[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
                    }
                )
                accu=accuracy_score(y_true=y_train[j * BATCH_SIZE:(j + 1) * BATCH_SIZE],y_pred=prediction)
                accus.append(accu)
                ls.append(l)

            print("Epoch:", i, "-loss:",sum(ls) / len(ls),"-accuracy:",sum(accus)/len(accus))

            #sava models
            saver.save(sess=sess,save_path=MODEL_SAVING_PATH,global_step=i)


if __name__=="__main__":
    train()


