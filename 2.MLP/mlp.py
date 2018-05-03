'''
two hidden layer and one output layer
'''

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

class MLP():
    def __init__(self,num_hidden1_unit,num_hidden2_unit):
        #basic environment
        self.graph=tf.Graph()
        self.session=tf.Session(graph=self.graph)

        #amount of neurals(units) in hidden layer 1 and 2
        self.n_h1=num_hidden1_unit
        self.n_h2=num_hidden2_unit

    #training
    def fit(self,X,y,epochs=1,batch_size=100,learning_rate=0.001,print_log=False):
        #num of samples,features and category
        n_samples=X.shape[0]
        n_features=X.shape[1]

        #one hot-encoding,num of category
        y_dummy=pd.get_dummies(data=y).values
        n_category=y_dummy.shape[1]

        # shuffle for random sampling
        sp = ShuffleSplit(n_splits=epochs, train_size=0.8)
        indices = sp.split(X=X)

        #############################define graph,and can be a forward process###########################
        with self.graph.as_default():
            #data place holder
            with tf.name_scope("Input"):
                self.X_p = tf.placeholder(dtype=tf.float32, shape=(None, n_features),name="X_p")
                self.y_dummy_p = tf.placeholder(dtype=tf.float32, shape=(None, n_category),name="y_dummy_p")
                self.y_p=tf.placeholder(dtype=tf.int64,shape=(None,),name="y_p")


            # -----------------------full connected layer 1(fc1)------------------------------#
            with tf.name_scope("FC1"):
                # weight(num_of_features x n_h1) and biases(n_h1,)
                self.weights_fc1 = tf.Variable(
                        initial_value=tf.truncated_normal(shape=(n_features, self.n_h1), stddev=0.1),
                        dtype=tf.float32,
                        name="weights_fc1"
                    )

                self.biases_fc1 = tf.Variable(
                        initial_value=tf.zeros(shape=(self.n_h1,)),
                        name="biases_fc1"
                    )
                # hidden layer 1 output(nxn_h1)
                self.activation_fc1 = tf.nn.relu(
                        features=tf.matmul(self.X_p, self.weights_fc1) + self.biases_fc1,
                        name="activation_fc1"
                    )
            #-----------------------------------------------------------------------------------


            # -----------------------full connected layer 2(fc2)------------------------------#
            with tf.name_scope("FC2"):
                # weight(n_h1 x n_h2) and biases(n_h2,)
                self.weights_fc2 = tf.Variable(
                        initial_value=tf.truncated_normal(shape=(self.n_h1, self.n_h2), stddev=0.1),
                        dtype=tf.float32,
                        name="weights_fc2"
                    )

                self.biases_fc2 = tf.Variable(
                        initial_value=tf.zeros(shape=(self.n_h2,)),
                        name="biases_fc2"
                    )

                # hidden layer 2 output(nxn_h2)
                self.activation_fc2 = tf.nn.relu(
                        features=tf.matmul(self.activation_fc1, self.weights_fc2) + self.biases_fc2,
                        name="activation_fc2"
                    )
            # -----------------------------------------------------------------------------------


            #-------------------------------output layer---------------------------------------------------
            with tf.name_scope("FC3"):
                # weights
                self.weights = tf.Variable(
                        initial_value=tf.zeros(shape=(self.n_h2, num_of_category)),
                        dtype=tf.float32,
                        name="weights"
                    )

                # biases
                self.biases = tf.Variable(
                        initial_value=tf.zeros(shape=(num_of_category,)),
                        name="biases"
                    )

                logits = tf.matmul(self.activation_fc2, self.weights) + self.biases
            #-------------------------------------------------------------------------------------------------------

            #probability
            self.prob=tf.nn.softmax(logits=logits,name="prob")
            #prediction
            self.pred=tf.argmax(input=self.prob,axis=1,name="pred")
            #accuracy
            self.accuracy=tf.reduce_mean(
                    input_tensor=tf.cast(x=tf.equal(x=self.pred,y=self.y_p),dtype=tf.float32),
                    name="accuracy"
                )
            #loss
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y_dummy_p))

            tf.summary.scalar(name="loss",tensor=self.cross_entropy)

            #optimizer
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cross_entropy)

            self.merged=tf.summary.merge_all()
            # visualization
            self.writer = tf.summary.FileWriter(logdir="./log", graph=self.graph)

            self.init = tf.global_variables_initializer()



        #SGD training
        epoch=1
        with self.session.as_default():
            #initialize all variables
            self.session.run(self.init)

            print("------------traing start-------------")
            for train_index,validation_index in indices:
                trainDataSize=train_index.shape[0]
                validationDataSize=validation_index.shape[0]
                print("epoch:",epoch)

                #average train loss and validation loss
                train_losses=[]
                validation_losses=[]

                #average taing accuracy and validation accuracy
                train_accus=[]
                validation_accus=[]


                #mini batch
                for i in range(0,(trainDataSize//batch_size)):
                    _,train_loss,train_accuracy,all_summary=self.session.run(
                                    fetches=[self.optimizer,self.cross_entropy,self.accuracy,self.merged],

                                    feed_dict={self.X_p:X[train_index[i*batch_size:(i+1)*batch_size]],
                                                self.y_dummy_p:y_dummy[train_index[i*batch_size:(i+1)*batch_size]],
                                                self.y_p:y[train_index[i*batch_size:(i+1)*batch_size]]
                                            }
                                    )
                    #add summary to event file
                    self.writer.add_summary(summary=all_summary,global_step=i)

                    validation_loss,validation_accuracy=self.session.run(
                                    fetches=[self.cross_entropy,self.accuracy],

                                    feed_dict={self.X_p:X[validation_index],
                                                self.y_dummy_p:y_dummy[validation_index],
                                                self.y_p:y[validation_index]
                                            }
                                    )

                    #add to list to compute average value
                    train_losses.append(train_loss)
                    validation_losses.append(validation_loss)
                    train_accus.append(train_accuracy)
                    validation_accus.append(validation_accuracy)


                    #weather print training infomation
                    if(print_log):
                        print("#############################################################")
                        print("batch: ",i*batch_size,"~",(i+1)*batch_size,"of epoch:",epoch)
                        print("training loss:",train_loss)
                        print("validation loss:",validation_loss)
                        print("train accuracy:", train_accuracy)
                        print("validation accuracy:", validation_accuracy)
                        print("#############################################################\n")

               # print("train_losses:",train_losses)
                ave_train_loss=sum(train_losses)/len(train_losses)
                ave_validation_loss=sum(validation_losses)/len(validation_losses)
                ave_train_accuracy=sum(train_accus)/len(train_accus)
                ave_validation_accuracy=sum(validation_accus)/len(validation_accus)
                print("average training loss:",ave_train_loss)
                print("average validation loss:",ave_validation_loss)
                print("average training accuracy:", ave_train_accuracy)
                print("average validation accuracy:", ave_validation_accuracy)

                epoch+=1


    def predict(self,X):
        with self.session.as_default():
            pred = self.session.run(fetches=self.pred, feed_dict={self.X_p: X})
        return pred

    def predict_prob(self,X):
        with self.session.as_default():
            prob=self.session.run(fetches=self.prob,feed_dict={self.X_p:X})
        return prob