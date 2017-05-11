'''
    only have one output layer
    the shape of weights and bias only define by the input features and output category
'''
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score


class Perceptron():
    def __init__(self):
        #basic environment
        self.graph=tf.Graph()
        self.session=tf.Session(graph=self.graph)


    #add some variables or constant etc to a graph
    def define_framewrok(self,num_of_features,num_of_category):
        with self.graph.as_default():
            #weights
            self.weights=tf.Variable(initial_value=tf.zeros(shape=(num_of_features,num_of_category)),name="weights")
            #biases
            self.biases=tf.Variable(initial_value=tf.zeros(shape=(num_of_category,)),name="biases")


    #forward compute
    def forward(self,X):
        with self.graph.as_default():
            logits=tf.matmul(X,self.weights)+self.biases
        return logits

    #training
    def fit(self,X,y,epochs=5,batch_size=100,print_log=False):
        #num of samples,features and category
        rows=X.shape[0]
        cols=X.shape[1]

        #one hot-encoding
        y_dummy=pd.get_dummies(data=y).values

        #how many category
        category=y_dummy.shape[1]

        #add op into graph
        self.define_framewrok(cols,category)

        #shuffle for random sampling
        sp=ShuffleSplit(n_splits=epochs,train_size=0.8)
        indices=sp.split(X=X)

        #placeholder
        with self.graph.as_default():
            X_p=tf.placeholder(dtype=tf.float32,shape=(None,cols))
            y_p=tf.placeholder(dtype=tf.float32,shape=(None,category))
            logits=self.forward(X_p)
           # pred=self.predict(X_p)
            cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_p))
            optimizer=tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

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
                    _,train_loss=self.session.run(fetches=[optimizer,cross_entropy],
                                          feed_dict={X_p:X[train_index[i*batch_size:(i+1)*batch_size]],y_p:y_dummy[train_index[i*batch_size:(i+1)*batch_size]]})
                    validation_loss=self.session.run(fetches=cross_entropy,
                                             feed_dict={X_p:X[validation_index],y_p:y_dummy[validation_index]})

                     #prediction in training process
                    #train_pred=self.predict(X=X[train_index[i*batch_size:(i+1)*batch_size]])
                    #validation_pred=self.predict(X=X[validation_index])

                    #accuracy in training process
                   # train_accuracy=self.accuracy(y_true=y[train_index[i*batch_size:(i+1)*batch_size]],y_pred=train_pred)
                   # validation_accuracy=self.accuracy(y_true=y[validation_index],y_pred=validation_pred)

                    #add to list to compute average value
                    train_losses.append(train_loss)
                    validation_losses.append(validation_loss)
                   # train_accus.append(train_accuracy)
                   # validation_accus.append(validation_accus)


                    #weather print training infomation
                    if(print_log):
                        print("training loss:",train_loss)
                        print("validation loss:",validation_loss)
                        print("train accuracy:", train_accuracy)
                        print("validation accuracy:", validation_accuracy)

               # print("train_losses:",train_losses)
                ave_train_loss=sum(train_losses)/len(train_losses)
                ave_validation_loss=sum(validation_losses)/len(validation_losses)
              #  ave_train_accuracy=sum(train_accus)/len(train_accus)
              #  ave_validation_accuracy=sum(validation_accus)/len(validation_accus)
                print("average training loss:",ave_train_loss)
                print("average validation loss:",ave_validation_loss)
              #  print("average training accuracy:", ave_train_accuracy)
              #  print("average validation accuracy:", ave_validation_accuracy)
                epoch+=1

    def predict(self,X):
        with self.graph.as_default():
            prob=self.predict_prob(X)
            pred=tf.argmax(input=prob,axis=1)

        with self.session.as_default():
            result=pred.eval()
            return result

    def predict_prob(self,X):
        prob=tf.nn.softmax(self.forward(X))
        return prob

    def accuracy(self,y_true,y_pred):
        score=accuracy_score(y_true=y_true,y_pred=y_pred)
        return score
