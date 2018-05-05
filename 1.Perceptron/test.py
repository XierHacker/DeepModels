import os
import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
import perceptron
from utility import preprocessing


MAX_EPOCH=10
BATCH_SIZE=20
LEARNING_RATE=0.001
MODEL_SAVING_PATH="./saved_models/model.ckpt-8"

#load data
X_train,y_train,X_valid,y_valid,X_test=preprocessing.load_mnist(path="../../data/mnist/")
train_samples=X_train.shape[0]
valid_samples=X_valid.shape[0]

def test():
    #data placeholder
    X_p=tf.placeholder(dtype=tf.float32,shape=(None,perceptron.INPUT_DIM),name="X_p")
    y_p=tf.placeholder(dtype=tf.int32,shape=(None,),name="y_p")
    y_hot_p=tf.one_hot(indices=y_p,depth=perceptron.OUTPUT_DIM)

    #inference
    model=perceptron.Perceptron()
    logits=model.forward(X_p,regularizer=None)           #[batch_size,10]
    pred=tf.argmax(input=logits,axis=-1)            #[batch_size,]

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y_hot_p, logits=logits)
    #Saver class
    saver=tf.train.Saver()

    with tf.Session() as sess:
        #restore
        saver.restore(sess=sess,save_path=MODEL_SAVING_PATH)

        #prediction
        l, prediction = sess.run(
            fetches=[loss, pred],
            feed_dict={X_p: X_train,y_p: y_train}
        )
        accu = accuracy_score(y_true=y_train, y_pred=prediction)
        print("-loss:", l, "-accuracy:", accu)


if __name__=="__main__":
    test()