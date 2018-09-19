import os
import sys
import time
sys.path.append("../")
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
import mlp
from utility import preprocessing

#no info
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

TRAIN_SIZE=preprocessing.getTFRecordsAmount(tfFile="../dataset/MNIST/mnist_train.tfrecords")
print("train_size:",TRAIN_SIZE)
MAX_EPOCH=10
BATCH_SIZE=20
LEARNING_RATE=0.001
MODEL_SAVING_PATH="./saved_models/model.ckpt"


# 定义解析和预处理函数
def _parse_data(example_proto):
    parsed_features = tf.parse_single_example(
        serialized=example_proto,
        features={
            "image_raw": tf.FixedLenFeature(shape=(784,), dtype=tf.float32),
            "label": tf.FixedLenFeature(shape=(), dtype=tf.int64)
        }
    )
    # get single feature
    image = parsed_features["image_raw"]
    image=image/255
    label = parsed_features["label"]
    # decode raw
    # image = tf.decode_raw(bytes=raw, out_type=tf.int64)
    #image = tf.reshape(tensor=raw, shape=[28, 28])
    return image, label



def train(tfrecords_list):
    # data placeholder
    X_p = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, 784), name="X_p")
    y_p = tf.placeholder(dtype=tf.int64, shape=(BATCH_SIZE,), name="y_p")
    y_hot_p = tf.one_hot(indices=y_p, depth=10)

    #----------------------------------------use dataset API--------------------------------------
    # 创建dataset对象
    dataset = tf.data.TFRecordDataset(filenames=tfrecords_list)
    # 使用map处理得到新的dataset
    dataset = dataset.map(map_func=_parse_data)
    dataset = dataset.batch(BATCH_SIZE).shuffle(buffer_size=2).repeat()

    # 创建迭代器
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    #use regularizer
    regularizer=tf.contrib.layers.l2_regularizer(0.005)

    #model
    model=mlp.MLP()
    logits=model.forward(X_p,regularizer)           #[batch_size,10]
    pred=tf.argmax(input=logits,axis=-1)            #[batch_size,]

    #accuracy
    correct_prediction=tf.equal(pred,y_p)
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    #L-2 loss
    l2_loss=tf.losses.get_regularization_loss()
    loss=tf.losses.softmax_cross_entropy(onehot_labels=y_hot_p,logits=logits)+l2_loss
    optimizer=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss=loss)

    init=tf.global_variables_initializer()

    #Saver class
    saver=tf.train.Saver()

    with tf.Session() as sess:

        sess.run(init)
        for i in range(MAX_EPOCH):
            start_time=time.time()
            ls = []
            accus=[]
            for j in range(TRAIN_SIZE // BATCH_SIZE):
                image_,label_=sess.run(next_element)
                _, l ,accu= sess.run(
                    fetches=[optimizer, loss, accuracy],
                    feed_dict={X_p:image_,y_p:label_}
                )
                accus.append(accu)
                ls.append(l)


            print("Epoch:", i, "-loss:",sum(ls) / len(ls),"-accuracy:",sum(accus)/len(accus))
            print("Spend: ", time.time() - start_time, " Seconds")
            #sava models
            saver.save(sess=sess,save_path=MODEL_SAVING_PATH,global_step=i)


if __name__=="__main__":
    train(tfrecords_list=["../dataset/MNIST/mnist_train.tfrecords"])