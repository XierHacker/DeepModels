import os
import sys
sys.path.append("..")
sys.path.append("../../")
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
import alex_net
from utility import preprocessing


BATCH_SIZE=20
LEARNING_RATE=0.001
MODEL_SAVING_PATH="./saved_models/model.ckpt-8"
TFRECORDS_PATH="../../data/DogsVsCats/dog_vs_cat_valid.tfrecords"

valid_size=200


def test():
    # data placeholder
    X_p = tf.placeholder(
        dtype=tf.float32,
        shape=(None, alex_net.INPUT_HIGHT, alex_net.INPUT_WEIGHT, 3),
        name="X_p"
    )
    y_p = tf.placeholder(dtype=tf.int32, shape=(None,), name="y_p")
    y_hot_p = tf.one_hot(indices=y_p, depth=alex_net.OUTPUT_DIM)

    # use dataset API
    batch = preprocessing.generate_dog_batch(
        tfrecords_path=TFRECORDS_PATH,
        batch_size=valid_size,
        is_train=False
    )

    #inference
    model=alex_net.AlexNet()
    logits=model.forward(X_p,regularizer=None)           #[batch_size,10]
    pred=tf.argmax(input=logits,axis=-1)            #[batch_size,]

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y_hot_p, logits=logits)
    #Saver class
    saver=tf.train.Saver()

    with tf.Session() as sess:
        #restore
        saver.restore(sess=sess,save_path=MODEL_SAVING_PATH)
        #get data
        images,labels=sess.run(batch)
        #prediction
        l, prediction = sess.run(fetches=[loss, pred],feed_dict={X_p: images,y_p: labels})
        accu = accuracy_score(y_true=labels, y_pred=prediction)
        print("-loss:", l, "-accuracy:", accu)

if __name__=="__main__":
    test()