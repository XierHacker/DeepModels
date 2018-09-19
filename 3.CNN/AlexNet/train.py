import os
import sys
sys.path.append("..")
sys.path.append("../../")
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
import alex_net
from utility import preprocessing

TRAIN_SIZE=preprocessing.getTFRecordsAmount(tfFile="../../dataset/Dogs_VS_Cats/dog_vs_cat_train.tfrecords")
print("train_size:",TRAIN_SIZE)

MAX_EPOCH=20
BATCH_SIZE=64
LEARNING_RATE=0.001
MODEL_SAVING_PATH="./saved_models/model.ckpt"


# 定义解析和预处理函数
def _parse_data(example_proto):
    parsed_features = tf.parse_single_example(
        serialized=example_proto,
        features={
            "image_raw": tf.FixedLenFeature(shape=[], dtype=tf.string),
            "label": tf.FixedLenFeature(shape=[], dtype=tf.int64)
        }
    )
    # get single feature
    raw = parsed_features["image_raw"]
    label = parsed_features["label"]
    # decode raw
    image = tf.decode_raw(bytes=raw, out_type=tf.uint8)
    image = tf.reshape(tensor=image, shape=(250, 250, 3))

    # flip
    image = tf.image.random_flip_left_right(image=image)
    # crop
    image = tf.image.resize_image_with_crop_or_pad(image=image, target_height=224, target_width=224)
    # trans to float
    image = tf.image.convert_image_dtype(image=image, dtype=tf.float32)
    return image, label


def train(tfrecords_list):
    #data placeholder
    X_p=tf.placeholder(dtype=tf.float32,shape=(None,224,224,3),name="X_p")
    y_p=tf.placeholder(dtype=tf.int64,shape=(None,),name="y_p")
    y_hot_p=tf.one_hot(indices=y_p,depth=2)

    # ----------------------------------------use dataset API--------------------------------------
    # 创建dataset对象
    dataset = tf.data.TFRecordDataset(filenames=tfrecords_list)
    # 使用map处理得到新的dataset
    dataset = dataset.map(map_func=_parse_data)
    dataset = dataset.batch(BATCH_SIZE).shuffle(buffer_size=2).repeat()

    # 创建迭代器
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    #-----------------------------------------------------------------------------------------------

    #use regularizer
    regularizer=tf.contrib.layers.l2_regularizer(0.005)
    #model
    model=alex_net.AlexNet()
    logits=model.forward(X_p,regularizer)           #[batch_size,10]
    pred=tf.argmax(input=logits,axis=-1)            #[batch_size,]

    #collect_list=tf.get_collection(key="regularized")
    #print("collect_list.shape",collect_list)
    loss=tf.losses.softmax_cross_entropy(onehot_labels=y_hot_p,logits=logits)+tf.add_n(inputs=tf.get_collection(key="regularized"))
    optimizer=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss=loss)

    init=tf.global_variables_initializer()

    #Saver class
    saver=tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(MAX_EPOCH):
            print("Epoch:",i+1)
            start_time=time.time()
            ls = []
            accus=[]
            for j in range(TRAIN_SIZE // BATCH_SIZE):
                images,labels=sess.run(next_element)
                _, l ,prediction= sess.run(fetches=[optimizer, loss, pred],feed_dict={X_p: images,y_p: labels})
                accu=accuracy_score(y_true=labels, y_pred=prediction)
                accus.append(accu)
                ls.append(l)

            end_time=time.time()
            print("spend ",(end_time-start_time)/60,"mins")
            print("--loss:",sum(ls) / len(ls),"--accuracy:",sum(accus)/len(accus))

            #sava models
            saver.save(sess=sess,save_path=MODEL_SAVING_PATH,global_step=i)


if __name__=="__main__":
    train(["../../dataset/Dogs_VS_Cats/dog_vs_cat_train.tfrecords"])