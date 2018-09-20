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

TEST_SIZE=preprocessing.getTFRecordsAmount(tfFile="../../dataset/Dogs_VS_Cats/dog_vs_cat_valid.tfrecords")//5
print("test_size:",TEST_SIZE)


MODEL_SAVINT_PATH="./saved_models/model.ckpt-8"


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

    #image = tf.image.random_flip_left_right(image=image)        # flip
    # crop
    image = tf.image.resize_image_with_crop_or_pad(image=image, target_height=224, target_width=224)
    # trans to float
    image = tf.image.convert_image_dtype(image=image, dtype=tf.float32)
    return image, label


def test(tfrecords_list):
    #data placeholder
    X_p=tf.placeholder(dtype=tf.float32,shape=(None,224,224,3),name="X_p")
    y_p=tf.placeholder(dtype=tf.int64,shape=(None,),name="y_p")
    y_hot_p=tf.one_hot(indices=y_p,depth=2)

    # ----------------------------------------use dataset API--------------------------------------
    # 创建dataset对象
    dataset = tf.data.TFRecordDataset(filenames=tfrecords_list)
    # 使用map处理得到新的dataset
    dataset = dataset.map(map_func=_parse_data)
    dataset = dataset.batch(TEST_SIZE)

    # 创建迭代器
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    #-----------------------------------------------------------------------------------------------

    #use regularizer
    regularizer=None
    #model
    model=alex_net.AlexNet()
    logits=model.forward(X_p,regularizer)           #[batch_size,10]
    pred=tf.argmax(input=logits,axis=-1)            #[batch_size,]

    # accuracy
    correct_prediction = tf.equal(pred, y_p)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    loss=tf.losses.softmax_cross_entropy(onehot_labels=y_hot_p,logits=logits)
    #optimizer=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss=loss)

    init=tf.global_variables_initializer()

    #Saver class
    saver=tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        # restore
        saver.restore(sess=sess, save_path=MODEL_SAVINT_PATH)
        start_time=time.time()
        images,labels=sess.run(next_element)
        l ,accu= sess.run(fetches=[loss, accuracy],feed_dict={X_p: images,y_p: labels})
        end_time=time.time()
        print("spend ",(end_time-start_time)/60,"mins")
        print("--loss:",l,"--accuracy:",accu)


if __name__=="__main__":
    test(["../../dataset/Dogs_VS_Cats/dog_vs_cat_valid.tfrecords"])