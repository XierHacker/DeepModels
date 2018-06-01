import time
import numpy as np
import tensorflow as tf

#you can modify this according your task
INPUT_HIGHT=224
INPUT_WEIGHT=224
#NUM_HIDDEN_1=300
#NUN_HIDDEN_2=100
OUTPUT_DIM=2


class AlexNet():
    def __init__(self):
        #basic environment
        #self.input_dim=INPUT_DIM
        #self.num_hidden_1=NUM_HIDDEN_1
        #self.num_hidden_2=NUN_HIDDEN_2
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

    #pytorch-like
    def conv2d(self,variable_scope,X,out_channels, kernel_size, stride, padding="SAME"):
        in_channels=X.get_shape()[-1].value
        with tf.variable_scope(variable_scope):
            #weights
            filter_weights=tf.get_variable(
                name="filter_weights",
                # [filter_height, filter_width, in_channels, out_channels]
                shape=(kernel_size[0],kernel_size[1],in_channels,out_channels),
                dtype=tf.float32,
                initializer=tf.initializers.truncated_normal(stddev=0.1)
            )
            #bias
            filter_bias=tf.get_variable(
                name="filter_bias",
                shape=(out_channels,),
                dtype=tf.float32,
                initializer=tf.initializers.constant()
            )
            #conv
            conv=tf.nn.conv2d(input=X,filter=filter_weights,strides=[0,stride[0],stride[1],0],padding=padding)
            feature_map=tf.nn.bias_add(value=conv,bias=filter_bias)
            return feature_map


    # pytorch-like
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
        #-------------------------------------conv1------------------------------------------------------------
        #conv
        logits_conv1=self.conv2d(variable_scope="conv1",X=X,out_channels=96,kernel_size=[11,11],stride=[4,4])
        #max pooling
        logits_conv1=tf.nn.max_pool(value=logits_conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID")
        #activate
        logits_conv1=tf.nn.relu(features=logits_conv1)
        print("logits_conv1.shape",logits_conv1.shape)

        # -------------------------------------conv2------------------------------------------------------------
        # conv
        logits_conv2 = self.conv2d(variable_scope="conv2", X=logits_conv1, out_channels=256, kernel_size=[5, 5], stride=[1, 1])
        # max pooling
        logits_conv2 = tf.nn.max_pool(value=logits_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
        # activate
        logits_conv2 = tf.nn.relu(features=logits_conv2)
        print("logits_conv2.shape", logits_conv2.shape)

        # -------------------------------------conv3------------------------------------------------------------
        # conv
        logits_conv3 = self.conv2d(variable_scope="conv3", X=logits_conv2, out_channels=384, kernel_size=[3, 3],stride=[1, 1])
        # activate
        logits_conv3 = tf.nn.relu(features=logits_conv3)
        print("logits_conv3.shape", logits_conv3.shape)

        # -------------------------------------conv4------------------------------------------------------------
        # conv
        logits_conv4 = self.conv2d(variable_scope="conv4", X=logits_conv3, out_channels=384, kernel_size=[3, 3],stride=[1, 1])
        # activate
        logits_conv4 = tf.nn.relu(features=logits_conv4)
        print("logits_conv4.shape", logits_conv4.shape)

        # -------------------------------------conv5------------------------------------------------------------
        # conv
        logits_conv5 = self.conv2d(variable_scope="conv5", X=logits_conv4, out_channels=256, kernel_size=[3, 3],stride=[1, 1])
        # max pooling
        logits_conv5 = tf.nn.max_pool(value=logits_conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
        # activate
        logits_conv5 = tf.nn.relu(features=logits_conv5)
        print("logits_conv5.shape", logits_conv5.shape)

        #reshape to connect to fully conected layer
        shape = logits_conv5.get_shape().as_list()
        h = shape[1]
        w = shape[2]
        c = shape[3]
        plat_logits_conv5 = tf.reshape(tensor=logits_conv5,shape=(-1, h * w * c))
        print(plat_logits_conv5.shape)

        # -------------------------------------FC1------------------------------------------------------------
        # fc
        logits_fc1=self.linear(
            variable_scope="FC1",X=plat_logits_conv5,
            weights_shape=(h*w*c,4096),regularizer=regularizer
        )
        # activate
        logits_fc1 = tf.nn.relu(features=logits_fc1)

        # -------------------------------------FC2------------------------------------------------------------
        # fc
        logits_fc2 = self.linear(
            variable_scope="FC2",X=logits_fc1,
            weights_shape=(4096, 4096),regularizer=regularizer
        )
        # activate
        logits_fc2 = tf.nn.relu(features=logits_fc2)

        # -------------------------------------FC3------------------------------------------------------------
        # fc
        logits_fc3 = self.linear(
            variable_scope="FC3",X=logits_fc2,
            weights_shape=(4096, self.output_dim),regularizer=regularizer
        )
        # activate
        logits_fc3 = tf.nn.softmax(logits=logits_fc3,axis=1)

        return logits_fc3


if __name__=="__main__":
    input=tf.random_normal(shape=(10,32,32,3))

    model=AlexNet()
    model.forward(X=input,regularizer=None)

