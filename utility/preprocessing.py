'''
    xierhacker  2018.5.1
'''
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def getTFRecordsAmount(tfFile):
    '''
    统计tfrecords中样本数量
    :param tfFile: 相应样本集的tfrecords文件
    :return:    样本总数量
    '''
    num = 0
    for record in tf.python_io.tf_record_iterator(tfFile):
        num += 1
    return num



def cifar2tfrecord(in_path,out_name,is_train):
    # mapping name to number
    mapping_dict = {
        "frog": 0,
        "truck": 1,
        "deer": 2,
        "automobile": 3,
        "bird": 4,
        "horse": 5,
        "ship": 6,
        "cat": 7,
        "airplane": 8,
        "dog": 9
    }

    print("Trans Pictures To TFRecords!")

    if is_train:
        train_labels_frame = pd.read_csv(filepath_or_buffer=in_path + "trainLabels.csv")
        writer_train = tf.python_io.TFRecordWriter(path=out_name)
        writer_valid = tf.python_io.TFRecordWriter(path=out_name)

        # training set
        for i in range(1, 45000 + 1):
            pic = mpimg.imread(fname=in_path + "train/" + str(i) + ".png")
            pic_raw = pic.tostring()
            kind = mapping_dict[train_labels_frame["label"][i - 1]]

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[pic_raw])),
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[kind]))
                    }
                )
            )
            writer_train.write(record=example.SerializeToString())
        writer_train.close()

        # validation set
        for i in range(45000 + 1, 50000 + 1):
            pic = mpimg.imread(fname=in_path + "train/" + str(i) + ".png")
            pic_raw = pic.tostring()
            kind = mapping_dict[train_labels_frame["label"][i - 1]]

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[pic_raw])),
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[kind]))
                    }
                )
            )
            writer_valid.write(record=example.SerializeToString())
        writer_valid.close()

    else:
        pass






