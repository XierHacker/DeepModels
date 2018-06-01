'''
    xierhacker  2018.5.1
'''

import numpy as np
import pandas as pd
import tensorflow as tf

def mnist2tfrecord(in_path,out_path):
    pass

def load_mnist(path):
    train_frame = pd.read_csv(path+"train.csv")[:40000]
    valid_frame = pd.read_csv(path + "train.csv")[40000:]
    test_frame = pd.read_csv(path+"test.csv")

    y_train = train_frame.pop(item="label").values
    #print(y_train.shape)
    y_valid = valid_frame.pop(item="label").values
    #print(y_valid.shape)

    # trans format
    X_train = train_frame.astype(np.float32).values
    X_valid = valid_frame.astype(np.float32).values
    X_test = test_frame.astype(np.float32).values

    return X_train,y_train,X_valid,y_valid,X_test

def generate_mnist_batch(X,y,batch_size):
    #dataset API
    dataset_train=tf.data.Dataset.from_tensor_slices(
        tensors=(X,y)
    ).repeat().batch(batch_size=batch_size).shuffle(buffer_size=2)
    #iterator
    iterator=dataset_train.make_one_shot_iterator()
    #get batch
    batch=iterator.get_next()
    return batch


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


if __name__=="__main__":
    load_mnist(path="../../data/mnist/")

