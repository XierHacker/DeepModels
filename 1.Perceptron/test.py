import os
import sys
sys.path.append("../")
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
import perceptron
from utility import preprocessing

os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'


TEST_SIZE=preprocessing.getTFRecordsAmount(tfFile="../dataset/MNIST/mnist_valid.tfrecords")
print("test_size:",TEST_SIZE)
MODEL_SAVING_PATH="./saved_models/model.ckpt-5"


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
    label = parsed_features["label"]
    # decode raw
    # image = tf.decode_raw(bytes=raw, out_type=tf.int64)
    #image = tf.reshape(tensor=raw, shape=[28, 28])
    return image, label



def test(tfrecords_list):
    # data placeholder
    X_p = tf.placeholder(dtype=tf.float32, shape=(None, 784), name="X_p")
    y_p = tf.placeholder(dtype=tf.int64, shape=(None,), name="y_p")
    y_hot_p = tf.one_hot(indices=y_p, depth=10)

    #----------------------------------------use dataset API--------------------------------------
    # 创建dataset对象
    dataset = tf.data.TFRecordDataset(filenames=tfrecords_list)
    # 使用map处理得到新的dataset
    dataset = dataset.map(map_func=_parse_data)
    dataset = dataset.batch(TEST_SIZE).shuffle(buffer_size=2).repeat()

    # 创建迭代器
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    #use regularizer==None
    regularizer=None

    #model
    model=perceptron.Perceptron()
    logits=model.forward(X_p,regularizer)           #[batch_size,10]
    pred=tf.argmax(input=logits,axis=-1)            #[batch_size,]

    #accuracy
    correct_prediction=tf.equal(pred,y_p)
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    loss=tf.losses.softmax_cross_entropy(onehot_labels=y_hot_p,logits=logits)
    #optimizer=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss=loss)

    init=tf.global_variables_initializer()

    #Saver class
    saver=tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        # restore
        saver.restore(sess=sess, save_path=MODEL_SAVING_PATH)
        image_, label_ = sess.run(next_element)
        l, accu = sess.run(
            fetches=[loss, accuracy],
            feed_dict={X_p: image_, y_p: label_}
        )

        print("-loss:",l,"-accuracy:",accu)


if __name__=="__main__":
    test(tfrecords_list=["../dataset/MNIST/mnist_valid.tfrecords"])