import tensorflow as tf
import numpy as np
import pandas as pd
import math
import time
from datetime import datetime
from sklearn.model_selection import ShuffleSplit

#convolution operation
def convolution(input, filter_size, strides, out_channel, name):
    in_channel = input.get_shape()[-1].value
    with tf.name_scope("name") as scope:
        #filter/kernel
        filter=tf.Variable(
            initial_value=tf.truncated_normal(shape=(filter_size[0],filter_size[1],in_channel,out_channel)),
            dtype=tf.float32,
            name="filter"
        )
        #biases
        biases=tf.Variable(
            initial_value=tf.truncated_normal(shape=(out_channel,)),
            dtype=tf.float32,
            name="biases"
        )
        #conv
        conv=tf.nn.conv2d(
            input=input,
            filter=filter,
            strides=[1,strides[0],strides[1],1],
            padding="SAME",
            name="conv"
        )
        featureMap=tf.nn.bias_add(value=conv,bias=biases)
        #activation
        activation=tf.nn.relu(features=featureMap,name="activation")
        return activation


#max pooing operation
def maxPooling(input,size,strides,padding,name):
    return tf.nn.max_pool(value=input,
                          ksize=[1,size[0],size[1],1],
                          strides=[1,strides[0],strides[1],1],
                          padding=padding,
                          name=name
                )

#fully connected layer operation
def fc(input,out_channel,activate,name):
    in_channel=input.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        weights=tf.Variable(
            initial_value=tf.truncated_normal(shape=(in_channel,out_channel)),
            dtype=tf.float32,
            name="weights"
        )

        biases=tf.Variable(
            initial_value=tf.truncated_normal(shape=(out_channel,)),
            dtype=tf.float32,
            name="biases"
        )
        logits=tf.matmul(input,weights)+biases
        if activate==False:
            return logits
        else:
            return tf.nn.relu(features=logits,name="activation")



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
            #------------------------------------------------------------------------------

            # ------------------------conv layer 1------------------------------------------
            self.activation_conv1=convolution(
                input=self.X_p,filter_size=[11,11],strides=[4,4],
                out_channel=64,name="conv1"
            )
            self.pool_activation_conv1=maxPooling(
                input=self.activation_conv1,size=[3,3],strides=[2,2],
                padding="VALID",name="pooing_cov1"
            )
            self.print_info(self.pool_activation_conv1)
            # ------------------------------------------------------------------------------

            # ------------------------conv layer 2------------------------------------------
            self.activation_conv2 = convolution(
                input=self.pool_activation_conv1, filter_size=[5, 5],strides=[1, 1],
                out_channel=192, name="conv2"
            )
            self.print_info(self.activation_conv2)

            self.pool_activation_conv2 = maxPooling(
                input=self.activation_conv2, size=[3, 3], strides=[2, 2],
                padding="VALID", name="pooing_cov2"
            )

            self.print_info(self.pool_activation_conv2)
            # ------------------------------------------------------------------------------

            # ------------------------conv layer 3------------------------------------------
            self.activation_conv3 = convolution(
                input=self.pool_activation_conv2, filter_size=[3, 3], strides=[1, 1],
                out_channel=348, name="conv3"
            )
            self.print_info(self.activation_conv3)
            # ------------------------------------------------------------------------------


            # ------------------------conv layer 4------------------------------------------
            self.activation_conv4 = convolution(
                input=self.activation_conv3, filter_size=[3, 3], strides=[1, 1],
                out_channel=256, name="conv4"
            )

            self.print_info(self.activation_conv4)
            # ------------------------------------------------------------------------------

            # ------------------------conv layer 5------------------------------------------
            self.activation_conv5 = convolution(
                input=self.activation_conv4, filter_size=[3, 3], strides=[1, 1],
                out_channel=256, name="conv5"
            )
            self.print_info(self.activation_conv5)

            self.pool_activation_conv5 = maxPooling(
                input=self.activation_conv5, size=[3, 3], strides=[2, 2],
                padding="VALID", name="pooing_cov5"
            )
            self.print_info(self.pool_activation_conv5)



            # ---------------------------------------------------------------------------------
            # reshape,because it will connect to fc layer
            shape = self.pool_activation_conv5.get_shape().as_list()
            h = shape[1]
            w = shape[2]
            c = shape[3]
            self.plat_activation_conv5 = tf.reshape(
                tensor=self.pool_activation_conv5,
                shape=(-1, h * w * c)
            )

            # -----------------------full connected layer 1(fc1)------------------------------#
            self.activation_fc1=fc(self.plat_activation_conv5,out_channel=4096,activate=True,name="FC1")

            # -----------------------full connected layer 2(fc2)------------------------------#
            self.activation_fc2 = fc(self.activation_fc1, out_channel=4096,activate=True, name="FC2")

            # -------------------------------output layer---------------------------------------------------
            self.logits = fc(self.activation_fc2, out_channel=num_of_category, activate=False, name="FC3")


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
            self.init_op=tf.global_variables_initializer()

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
            self.session.run(self.init_op)



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

                #average time uses
                back_time=[]
                forward_time=[]

                # mini batch
                for i in range(0, (trainDataSize // batch_size)):
                    start_time_back = time.time()
                    _=self.session.run(
                        fetches=self.optimizer,
                        feed_dict={self.X_p: X[train_index[i * batch_size:(i + 1) * batch_size]],
                                   self.y_dummy_p: y_dummy[train_index[i * batch_size:(i + 1) * batch_size]],
                                   self.y_p: y[train_index[i * batch_size:(i + 1) * batch_size]]
                                   }
                    )
                    duration_back=time.time()-start_time_back


                    start_time_forward=time.time()
                    train_loss, train_accuracy = self.session.run(
                        fetches=[self.cross_entropy, self.accuracy],
                        feed_dict={self.X_p: X[train_index[i * batch_size:(i + 1) * batch_size]],
                                   self.y_dummy_p: y_dummy[train_index[i * batch_size:(i + 1) * batch_size]],
                                   self.y_p: y[train_index[i * batch_size:(i + 1) * batch_size]]
                                   }
                    )
                    duration_forward = time.time() - start_time_forward

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
                    back_time.append(duration_back)
                    forward_time.append(duration_forward)


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

                print("average training loss:", sum(train_losses) / len(train_losses))
                print("average validation loss:", sum(validation_losses) / len(validation_losses))
                print("average training accuracy:", sum(train_accus) / len(train_accus))
                print("average validation accuracy:", sum(validation_accus) / len(validation_accus))
                print("total back time uses per minibach/epoch:",sum(back_time)/len(back_time),"/",sum(back_time))
                print("total forward time uses per minibach/epoch:", sum(forward_time)/len(forward_time), "/", sum(forward_time))

                epoch += 1

    def predict(self, X):
        with self.session.as_default():
            pred = self.session.run(fetches=self.pred, feed_dict={self.X_p: X})
        return pred

    def predict_prob(self, X):
        with self.session.as_default():
            prob = self.session.run(fetches=self.prob, feed_dict={self.X_p: X})
        return prob