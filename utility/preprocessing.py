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

def cifar2tfrecord(in_path,out_path):
    pass



if __name__=="__main__":
    load_mnist(path="../../data/mnist/")

