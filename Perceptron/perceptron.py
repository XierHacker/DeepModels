'''
    only one output layer,no hidden layer
    the shape of weights and bias only define by the input features and output category
'''

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

FLAGS=tf.app.flags.FLAGS


class Perceptron():
    def __init__(self):
        #basic environment
        self.graph=tf.Graph()
        self.session=tf.Session(graph=self.graph)

    #contains forward and training
    def fit(self,X,y,epochs=5,batch_size=100,learning_rate=0.001,print_log=False):
        #num of samples,features and category
        n_samples=X.shape[0]
        n_features=X.shape[1]

        #one hot-encoding,num of category
        y_dummy=pd.get_dummies(data=y).values
        n_category=y_dummy.shape[1]

        # shuffle for random sampling
        sp = ShuffleSplit(n_splits=epochs, train_size=0.8)
        indices = sp.split(X=X)

        # best accuracy on validation test
        best_validation_accus = 0

        #epoch record
        epoch = 1

        ##########################define graph(forward computation)#####################
        with self.graph.as_default():
            #data place holder
            self.X_p = tf.placeholder(dtype=tf.float32,
                                      shape=(None, n_features),
                                      name="input_placeholder")

            self.y_dummy_p = tf.placeholder(dtype=tf.float32,
                                            shape=(None, n_category),
                                            name="label_dummy_placeholder")

            self.y_p=tf.placeholder(dtype=tf.int64,
                                    shape=(None,),
                                    name="label_placeholder")

            #--------------------------fully connected layer-----------------------------------#
            #weights(initialized to 0)
            self.weights=tf.Variable(initial_value=tf.zeros(shape=(n_features,n_category)),
                                     name="weights")

            #biases(initialized to 0)
            self.biases=tf.Variable(initial_value=tf.zeros(shape=(n_category,)),
                                    name="biases")

            #shape of logits is (None,num_of_category)
            logits = tf.matmul(self.X_p, self.weights) + self.biases

            #----------------------------------------------------------------------------------#

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
            self.cross_entropy = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y_dummy_p)
                )

            #optimizer
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cross_entropy)

            self.init = tf.global_variables_initializer()

        #SGD training
        with self.session as sess:
            sess.run(self.init)

            #restore
            #new_saver = tf.train.import_meta_graph('./model/my-model-10000.meta')
            #new_saver.restore(sess, './model/my-model-10000')

            print("------------traing start-------------")

            for train_index,validation_index in indices:
                print("epoch:", epoch)

                trainDataSize=train_index.shape[0]
                validationDataSize=validation_index.shape[0]

                #average train loss and validation loss
                train_losses=[]; validation_losses=[]
                #average taing accuracy and validation accuracy
                train_accus=[]; validation_accus=[]

                #mini batch
                for i in range(0,(trainDataSize//batch_size)):
                    _,train_loss,train_accuracy=self.session.run(
                                    fetches=[self.optimizer,self.cross_entropy,self.accuracy],

                                    feed_dict={self.X_p:X[train_index[i*batch_size:(i+1)*batch_size]],
                                                self.y_dummy_p:y_dummy[train_index[i*batch_size:(i+1)*batch_size]],
                                                self.y_p:y[train_index[i*batch_size:(i+1)*batch_size]]
                                            }
                                    )

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

                #when we get a new best validation accuracy,we store the model
                if best_validation_accus<ave_validation_accuracy:
                    print("we got a new best accuracy on validation set!")
                    # Creates a saver.
                    saver = tf.train.Saver()
                    saver.save(sess, './model/my-model-10000')

                    # Generates MetaGraphDef.
                    saver.export_meta_graph('./model/my-model-10000.meta')


    def predict(self,X):
        with self.session as sess:
            #restore model
            new_saver = tf.train.import_meta_graph('./model/my-model-10000.meta',clear_devices=True)
            new_saver.restore(sess, './model/my-model-10000')

            graph=tf.get_default_graph()
            pred=graph.get_operation_by_name("pred").outputs[0]
            X_p=graph.get_operation_by_name("input_placeholder").outputs[0]
            pred = sess.run(fetches=pred, feed_dict={X_p: X})
        return pred

    def predict_prob(self,X):
        with self.session.as_default():
            prob=self.session.run(fetches=self.prob,feed_dict={self.X_p:X})
        return prob
