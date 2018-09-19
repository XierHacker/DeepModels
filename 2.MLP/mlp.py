'''
    2 hidden layer
'''

import numpy as np
import tensorflow as tf

#you can modify this according your task
INPUT_DIM=784
NUM_HIDDEN_1=50
NUN_HIDDEN_2=50
OUTPUT_DIM=10


class MLP():
    def __init__(self):
        #basic environment
        self.input_dim=INPUT_DIM
        self.num_hidden_1=NUM_HIDDEN_1
        self.num_hidden_2=NUN_HIDDEN_2
        self.output_dim=OUTPUT_DIM


    def forward(self,X,regularizer):
        logits_fc1=tf.layers.dense(
            inputs=X,
            units=self.num_hidden_1,
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.initializers.constant(0.1),
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
            #activity_regularizer=regularizer,
            trainable=True,
            name="logits_fc1",
            reuse=None
        )
        logits_fc2 = tf.layers.dense(
            inputs=logits_fc1,
            units=self.num_hidden_2,
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.initializers.constant(0.1),
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
            # activity_regularizer=regularizer,
            trainable=True,
            name="logits_fc2",
            reuse=None
        )

        logits_fc3 = tf.layers.dense(
            inputs=logits_fc2,
            units=self.output_dim,
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.initializers.constant(0.1),
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
            # activity_regularizer=regularizer,
            trainable=True,
            name="logits_fc3",
            reuse=None
        )

        return logits_fc3














'''
#-----------------------------------------Old Version-----------------------------------#
class MLP():
    def __init__(self):
        #basic environment
        self.input_dim=INPUT_DIM
        self.num_hidden_1=NUM_HIDDEN_1
        self.num_hidden_2=NUN_HIDDEN_2
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

    # pytorch API like
    def linear(self, variable_scope, X, weights_shape, regularizer):
        with tf.variable_scope(variable_scope):
            weights = self.get_weights_variable(
                shape=weights_shape,
                regularizer=regularizer
            )

            biases = tf.get_variable(
                name="biases",
                shape=(weights_shape[1],),
                dtype=tf.float32,
                initializer=tf.initializers.constant()
            )
            logits = tf.matmul(X, weights) + biases
            return logits

    def forward(self,X,regularizer):
        logits_fc1 = self.linear(
            variable_scope="fc1",
            X=X,
            weights_shape=(self.input_dim,self.num_hidden_1),
            regularizer=regularizer
        )
        logits_fc2 = self.linear(
            variable_scope="fc2",
            X=logits_fc1,
            weights_shape=(self.num_hidden_1, self.num_hidden_2),
            regularizer=regularizer
        )
        logits_fc3 = self.linear(
            variable_scope="fc3",
            X=logits_fc2,
            weights_shape=(self.num_hidden_2, self.output_dim),
            regularizer=regularizer
        )
        return logits_fc3



if __name__=="__main__":
    model=Perceptron()
    weights=model.get_weights_variable(
        shape=(2,2),
        regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001)
    )
    print(weights)
    re=tf.get_collection(key="regularized")
    print(re)
'''
