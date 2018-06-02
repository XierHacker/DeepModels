'''
    xierhacker  2018.5.1
'''
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

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

def dog2tfrecords(folder,out_path,is_train):
    '''
        :param folder:  folder to storage pics
        :param is_train:train set of validation set
        :return:
        '''
    print("Trans Pictures To TFRecords")
    if is_train:
        kind_cat = 0
        kind_dog = 1
        writer_train = tf.python_io.TFRecordWriter(path=out_path+"dog_vs_cat_train.tfrecords")
        writer_valid = tf.python_io.TFRecordWriter(path=out_path+"dog_vs_cat_valid.tfrecords")

        # training set
        for i in range(10000):
            pic_cat = cv2.imread(filename=folder + "train/cat." + str(i) + ".jpg", flags=cv2.IMREAD_UNCHANGED)
            # resize to 250x250
            pic_cat = cv2.resize(src=pic_cat, dsize=(250, 250), interpolation=cv2.INTER_AREA)
            # to string
            pic_cat_raw = pic_cat.tostring()
            #print(pic_cat.shape)

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[pic_cat_raw])),
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[kind_cat]))
                    }
                )
            )
            writer_train.write(record=example.SerializeToString())

            pic_dog = cv2.imread(filename=folder + "train/dog." + str(i) + ".jpg", flags=cv2.IMREAD_UNCHANGED)
            # resize to 250x250
            pic_dog = cv2.resize(src=pic_dog, dsize=(250, 250), interpolation=cv2.INTER_AREA)
            # to string
            pic_dog_raw = pic_dog.tostring()
            #print(pic_dog.shape)

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[pic_dog_raw])),
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[kind_dog]))
                    }
                )
            )
            writer_train.write(record=example.SerializeToString())

        writer_train.close()

        # validation set
        for i in range(10000, 12500):
            pic_cat = cv2.imread(filename=folder + "train/cat." + str(i) + ".jpg", flags=cv2.IMREAD_UNCHANGED)
            # resize to 300x300
            pic_cat = cv2.resize(src=pic_cat, dsize=(250, 250), interpolation=cv2.INTER_AREA)
            # to string
            pic_cat_raw = pic_cat.tostring()
            #print(pic_cat.shape)

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[pic_cat_raw])),
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[kind_cat]))
                    }
                )
            )
            writer_valid.write(record=example.SerializeToString())

            pic_dog = cv2.imread(filename=folder + "train/dog." + str(i) + ".jpg", flags=cv2.IMREAD_UNCHANGED)
            # resize to 300x300
            pic_dog = cv2.resize(src=pic_dog, dsize=(250, 250), interpolation=cv2.INTER_AREA)
            # to string
            pic_dog_raw = pic_dog.tostring()
            #print(pic_dog.shape)

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[pic_dog_raw])),
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[kind_dog]))
                    }
                )
            )
            writer_valid.write(record=example.SerializeToString())

        writer_valid.close()

    else:
        pass


def generate_dog_batch(tfrecords_path,batch_size):
    # tfrecord 文件列表
    file_list = [tfrecords_path]

    # 创建dataset对象
    dataset = tf.data.TFRecordDataset(filenames=file_list)

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
        # crop
        image = tf.image.resize_image_with_crop_or_pad(image=image, target_height=224, target_width=224)
        # trans to float
        image = tf.image.convert_image_dtype(image=image, dtype=tf.float32)
        return image, label

    # 使用map处理得到新的dataset
    dataset = dataset.map(map_func=_parse_data)
    # 使用batch_size为32生成mini-batch
    dataset = dataset.repeat().batch(batch_size=batch_size).shuffle(buffer_size=2)
    # 创建迭代器
    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()
    return batch


if __name__=="__main__":
    #dog2tfrecords(folder="../../data/DogsVsCats/",out_path="../../data/DogsVsCats/",is_train=True)
    batch=generate_dog_batch(tfrecords_path="../../data/DogsVsCats/dog_vs_cat_train.tfrecords",batch_size=2)
    with tf.Session() as sess:
        for i in range(10):
            image, label = sess.run(batch)
            print("label:", label)
            print("image.shape:", image.shape)
            print("label.shape", label.shape)
            for j in range(2):
                plt.imshow(image[j])
                plt.show()






