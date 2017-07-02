'''



'''
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import time
from datetime import datetime
from sklearn.model_selection import ShuffleSplit


class AlexNet():
    def __init__(self):
        # basic environment
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

    #use to debug
    def print_info(self,t):
        print(t.op.name," ",t.get_shape().as_list)


    # the main framework of this model
    def define_framewrok(self, in_height, in_width, in_channels, num_of_category):
        with self.graph.as_default():
            # ---------------------data place holder----------------------------------------
            self.X_p = tf.placeholder(
                dtype=tf.float32,
                shape=(None, in_height, in_width, in_channels),
                name="X_p"
            )
            self.y_dummy_p = tf.placeholder(
                dtype=tf.float32,
                shape=(None, num_of_category),
                name="y_dummy_p"
            )
            self.y_p = tf.placeholder(
                dtype=tf.int64,
                shape=(None,),
                name="y_p"
            )
            #-------------------------------------------------------------------------------

            parameters=[]

            # ------------------------conv layer 1------------------------------------------
            with tf.name_scope("conv1") as scope:
                self.filter_conv1 = tf.Variable(
                    initial_value=tf.truncated_normal(shape=(11, 11, in_channels, 64)),
                    dtype=tf.float32,
                    name="fiter_conv1"
                )

                self.biases_conv1 = tf.Variable(
                    initial_value=tf.truncated_normal(shape=(64,)),
                    dtype=tf.float32,
                    name="biases_conv1"
                )

                # we get (n x height x width x 64) feature map
                self.featureMap_conv1 = tf.nn.conv2d(
                    input=self.X_p,
                    filter=self.filter_conv1,
                    strides=[1, 4, 4, 1],
                    padding="SAME",
                    name="FeatureMap_conv1") + self.biases_conv1

                #we drop the LRN layer,because it's seems not help in our tesk

                # pooing --> (n x height/2 x width/2  x 6)
                self.pool_featureMap_conv1 = tf.nn.max_pool(
                    value=self.featureMap_conv1,
                    ksize=[1, 3, 3, 1],
                    strides=[1, 2, 2, 1],
                    padding="VALID",
                    name="pooing_conv1"
                )
                # activation as next layer's input (n x height/2 x width/2  x 6)
                self.activation_conv1 = tf.nn.relu(features=self.pool_featureMap_conv1, name="relu_conv1")

            # ------------------------------------------------------------------------------

            # ------------------------conv layer 2------------------------------------------
            with tf.name_scope("conv2") as scope:
                self.filter_conv2 = tf.Variable(
                    initial_value=tf.truncated_normal(shape=(5, 5, 64, 192)),
                    dtype=tf.float32,
                    name="fiter_conv2"
                )

                self.biases_conv2 = tf.Variable(
                    initial_value=tf.truncated_normal(shape=(192,)),
                    dtype=tf.float32,
                    name="biases_conv2"
                )

                # we get (n x height x width x 192) feature map
                self.featureMap_conv2 = tf.nn.conv2d(
                    input=self.activation_conv1,
                    filter=self.filter_conv2,
                    strides=[1, 1, 1, 1],
                    padding="SAME",
                    name="FeatureMap_conv2") + self.biases_conv2

                # we drop the LRN layer,because it's seems not help in our tesk

                # pooing --> (n x height/2 x width/2  x 6)
                self.pool_featureMap_conv2 = tf.nn.max_pool(
                    value=self.featureMap_conv2,
                    ksize=[1, 3, 3, 1],
                    strides=[1, 2, 2, 1],
                    padding="VALID",
                    name="pooing_conv2"
                )
                # activation as next layer's input (n x height/2 x width/2  x 6)
                self.activation_conv2 = tf.nn.relu(features=self.pool_featureMap_conv2, name="relu_conv2")

            # ------------------------------------------------------------------------------

            # ------------------------conv layer 3------------------------------------------
            with tf.name_scope("conv3") as scope:
                self.filter_conv3 = tf.Variable(
                    initial_value=tf.truncated_normal(shape=(3, 3, 192, 384)),
                    dtype=tf.float32,
                    name="fiter_conv3"
                )

                self.biases_conv3 = tf.Variable(
                    initial_value=tf.truncated_normal(shape=(384,)),
                    dtype=tf.float32,
                    name="biases_conv3"
                )

                # we get (n x height x width x 192) feature map
                self.featureMap_conv3 = tf.nn.conv2d(
                    input=self.activation_conv2,
                    filter=self.filter_conv3,
                    strides=[1, 1, 1, 1],
                    padding="SAME",
                    name="FeatureMap_conv3") + self.biases_conv3


                # there has no pooing

                # activation as next layer's input (n x height/2 x width/2  x 6)
                self.activation_conv3 = tf.nn.relu(features=self.featureMap_conv3, name="relu_conv3")

            # ------------------------------------------------------------------------------

            # ------------------------conv layer 4------------------------------------------
            with tf.name_scope("conv4") as scope:
                self.filter_conv4 = tf.Variable(
                    initial_value=tf.truncated_normal(shape=(3, 3, 384, 256)),
                    dtype=tf.float32,
                    name="fiter_conv4"
                )

                self.biases_conv4 = tf.Variable(
                    initial_value=tf.truncated_normal(shape=(256,)),
                    dtype=tf.float32,
                    name="biases_conv4"
                )

                # we get (n x height x width x 256) feature map
                self.featureMap_conv4 = tf.nn.conv2d(
                    input=self.activation_conv3,
                    filter=self.filter_conv4,
                    strides=[1, 1, 1, 1],
                    padding="SAME",
                    name="FeatureMap_conv3") + self.biases_conv4

                # there has no pooing

                # activation as next layer's input (n x height/2 x width/2  x 6)
                self.activation_conv4 = tf.nn.relu(features=self.featureMap_conv4, name="relu_conv4")

            # ------------------------------------------------------------------------------


            # ------------------------conv layer 5------------------------------------------
            with tf.name_scope("conv5") as scope:
                self.filter_conv5 = tf.Variable(
                    initial_value=tf.truncated_normal(shape=(3, 3, 256, 256)),
                    dtype=tf.float32,
                    name="fiter_conv5"
                )

                self.biases_conv5 = tf.Variable(
                    initial_value=tf.truncated_normal(shape=(256,)),
                    dtype=tf.float32,
                    name="biases_conv5"
                )

                # we get (n x height x width x 256) feature map
                self.featureMap_conv5 = tf.nn.conv2d(
                    input=self.activation_conv4,
                    filter=self.filter_conv5,
                    strides=[1, 1, 1, 1],
                    padding="SAME",
                    name="FeatureMap_conv5") + self.biases_conv5

                # we drop the LRN layer,because it's seems not help in our tesk

                # pooing --> (n x height/2 x width/2  x 6)
                self.pool_featureMap_conv5 = tf.nn.max_pool(
                    value=self.featureMap_conv5,
                    ksize=[1, 3, 3, 1],
                    strides=[1, 2, 2, 1],
                    padding="VALID",
                    name="pooing_conv5"
                )
                # activation as next layer's input (n x height/2 x width/2  x 6)
                self.activation_conv5 = tf.nn.relu(features=self.pool_featureMap_conv5, name="relu_conv5")


            # ---------------------------------------------------------------------------------
            # reshape,because it will connect to fc layer
            shape = self.activation_conv5.get_shape().as_list()
            h = shape[1]
            w = shape[2]
            c = shape[3]
            self.plat_activation_conv5 = tf.reshape(
                tensor=self.activation_conv5,
                shape=(-1, h * w * c)
            )

            # -----------------------full connected layer 1(fc1)------------------------------#
            with tf.name_scope("FC1") as scope:
                # weight(num_of_features x 4096) and biases(4096,)
                self.weights_fc1 = tf.Variable(
                    initial_value=tf.truncated_normal(shape=(h * w * c, 4096), stddev=0.1),
                    dtype=tf.float32,
                    name="weights_fc1"
                )
                self.biases_fc1 = tf.Variable(
                    initial_value=tf.zeros(shape=(4096,)),
                    name="biases_fc1"
                )

                # FC1 output(nx4096)
                self.activation_fc1 = tf.nn.relu(
                    features=tf.matmul(self.plat_activation_conv5, self.weights_fc1) + self.biases_fc1,
                    name="activation_fc1"
                )
            # -----------------------------------------------------------------------------------



            # -----------------------full connected layer 2(fc2)------------------------------#
            with tf.name_scope("FC2") as scope:
                # weight(4096 x 4096) and biases(4096,)
                self.weights_fc2 = tf.Variable(
                    initial_value=tf.truncated_normal(shape=(4096, 4096), stddev=0.1),
                    dtype=tf.float32,
                    name="weights_fc2"
                )
                self.biases_fc2 = tf.Variable(
                    initial_value=tf.zeros(shape=(4096,)),
                    name="biases_fc2"
                )

                # hidden layer 2 output(nx4096)
                self.activation_fc2 = tf.nn.relu(
                    features=tf.matmul(self.activation_fc1, self.weights_fc2) + self.biases_fc2,
                    name="activation_fc2"
                )
            # -----------------------------------------------------------------------------------


            # -------------------------------output layer---------------------------------------------------
            with tf.name_scope("OutPut") as scope:
                # weights(4096,1000)
                self.weights = tf.Variable(
                    initial_value=tf.zeros(shape=(4096, num_of_category)),
                    dtype=tf.float32,
                    name="weights"
                )
                # biases(1000,)
                self.biases = tf.Variable(
                    initial_value=tf.zeros(shape=(num_of_category,)),
                    name="biases"
                )

                self.logits = tf.matmul(self.activation_fc2, self.weights) + self.biases
            # -------------------------------------------------------------------------------------------------------



            # probability
            self.prob = tf.nn.softmax(logits=self.logits, name="prob")
            # prediction
            self.pred = tf.argmax(input=self.prob, axis=1, name="pred")
            # accuracy
            self.accuracy = tf.reduce_mean(
                input_tensor=tf.cast(x=tf.equal(x=self.pred, y=self.y_p),
                                     dtype=tf.float32),
                name="accuracy"
            )
            # loss
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_dummy_p))
            # optimizer
            self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cross_entropy)

            self.init = tf.global_variables_initializer()

    # training
    def fit(self, X, y, epochs=20, batch_size=500, print_log=False):
        # num of samples,features and category
        n_samples = X.shape[0]
        height = X.shape[1]
        width = X.shape[2]
        channels = X.shape[3]

        # one hot-encoding,num of category
        y_dummy = pd.get_dummies(data=y).values
        n_category = y_dummy.shape[1]

        # add op into graph
        self.define_framewrok(height, width, channels, n_category)

        # shuffle for random sampling
        sp = ShuffleSplit(n_splits=epochs, train_size=0.8)
        indices = sp.split(X=X)

        # SGD training
        epoch = 1
        with self.session.as_default():
            # initialize all variables
            self.session.run(self.init)

            print("------------traing start-------------")
            for train_index, validation_index in indices:
                trainDataSize = train_index.shape[0]
                validationDataSize = validation_index.shape[0]
                print("epoch:", epoch)

                # average train loss and validation loss
                train_losses = []
                validation_losses = []

                # average taing accuracy and validation accuracy
                train_accus = []
                validation_accus = []

                # mini batch
                for i in range(0, (trainDataSize // batch_size)):
                    _, train_loss, train_accuracy = self.session.run(
                        fetches=[self.optimizer, self.cross_entropy, self.accuracy],
                        feed_dict={self.X_p: X[train_index[i * batch_size:(i + 1) * batch_size]],
                                   self.y_dummy_p: y_dummy[train_index[i * batch_size:(i + 1) * batch_size]],
                                   self.y_p: y[train_index[i * batch_size:(i + 1) * batch_size]]
                                   }
                    )

                    validation_loss, validation_accuracy = self.session.run(
                        fetches=[self.cross_entropy, self.accuracy],
                        feed_dict={self.X_p: X[validation_index],
                                   self.y_dummy_p: y_dummy[validation_index],
                                   self.y_p: y[validation_index]
                                   }
                    )

                    # add to list to compute average value
                    train_losses.append(train_loss)
                    validation_losses.append(validation_loss)
                    train_accus.append(train_accuracy)
                    validation_accus.append(validation_accuracy)

                    # weather print training infomation
                    if (print_log):
                        print("#############################################################")
                        print("batch: ", i * batch_size, "~", (i + 1) * batch_size, "of epoch:", epoch)
                        print("training loss:", train_loss)
                        print("validation loss:", validation_loss)
                        print("train accuracy:", train_accuracy)
                        print("validation accuracy:", validation_accuracy)
                        print("#############################################################\n")

                        # print("train_losses:",train_losses)
                ave_train_loss = sum(train_losses) / len(train_losses)
                ave_validation_loss = sum(validation_losses) / len(validation_losses)
                ave_train_accuracy = sum(train_accus) / len(train_accus)
                ave_validation_accuracy = sum(validation_accus) / len(validation_accus)
                print("average training loss:", ave_train_loss)
                print("average validation loss:", ave_validation_loss)
                print("average training accuracy:", ave_train_accuracy)
                print("average validation accuracy:", ave_validation_accuracy)
                epoch += 1

    def predict(self, X):
        with self.session.as_default():
            pred = self.session.run(fetches=self.pred, feed_dict={self.X_p: X})
        return pred

    def predict_prob(self, X):
        with self.session.as_default():
            prob = self.session.run(fetches=self.prob, feed_dict={self.X_p: X})
        return prob