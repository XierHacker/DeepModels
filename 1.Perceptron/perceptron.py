'''
    1.no hidden layer
    2.the shape of weights and bias only define by the input features and output categories
'''

import numpy as np
import tensorflow as tf

#you can modify this according your task
INPUT_DIM=784
OUTPUT_DIM=10

class Perceptron():
    def __init__(self):
        #basic environment
        self.input_dim=INPUT_DIM
        self.output_dim=OUTPUT_DIM

    def get_weights_variable(self,shape,regularizer):
        weights=tf.get_variable(
            name="weights",
            shape=shape,
            dtype=tf.float32,
            initializer=tf.initializers.truncated_normal(stddev=0.1)
        )

        if regularizer!=None:
            tf.add_to_collection(name="regularized",value=regularizer(weights))
        return weights


    def forward(self,X,regularizer):
        with tf.variable_scope("layer1"):
            weights = self.get_weights_variable(
                shape=(self.input_dim, self.output_dim),
                regularizer=regularizer
            )
            biases = tf.get_variable(
                name="biases",
                shape=(self.output_dim,),
                dtype=tf.float32,
                initializer=tf.initializers.constant()
            )
            logits = tf.matmul(X, weights) + biases
            return logits


if __name__=="__main__":
    model=Perceptron()
    weights=model.get_weights_variable(
        shape=(2,2),
        regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001)
    )
    print(weights)
    re=tf.get_collection(key="regularized")
    print(re)




