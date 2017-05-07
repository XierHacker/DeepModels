import tensorflow as tf
from sklearn.model_selection import ShuffleSplit
import numpy as np

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
            self.init=tf.global_variables_initializer()

    #forward compute
    def forward(self,X):
        with self.graph.as_default():
            logits=tf.matmul(X,self.weights)+self.biases
            prob=tf.nn.softmax(logits)
        return prob

    #training
    def fit(self,X,y,epochs=10,batch_size=200,print_log=True):
        #num of samples,features and category
        rows=X.shape[0]
        cols=X.shape[1]
        category=y.shape[1]

        #add op into graph
        self.define_framewrok(cols,category)

        #shuffle
        sp=ShuffleSplit(n_splits=epochs,train_size=0.8)
        indices=sp.split(X=X)
        #placeholder
        with self.graph.as_default():
            X_p=tf.placeholder(dtype=tf.float32,shape=(None,cols))
            y_p=tf.placeholder(dtype=tf.float32,shape=(None,category))
            prob=self.forward(X_p)
            cross_ectropy=tf.reduce_mean(-tf.reduce_sum(input_tensor=y_p*tf.log(prob),reduction_indices=1))
            optimizer=tf.train.GradientDescentOptimizer(0.05).minimize(cross_ectropy)



        #SGD training
        epoch=1
        with self.session as sess:
            sess.run(self.init)
            print("------------traing-------------")
            for train_index,test_index in indices:
                print("epoch:",epoch)
                #mini batch
                for i in range(0,rows,batch_size):
                    _,loss=sess.run(fetches=[optimizer,cross_ectropy],
                                    feed_dict={X_p:X[train_index[i:i+batch_size]],y_p:y[train_index[i:i+batch_size]]})
                    print("loss:",loss)

                epoch+=1



    def predict(self,X):
        pass

    def predict_prob(self,X):
        pass