import time
import numpy as np
import tensorflow as tf

#you can modify this according your task
INPUT_HIGHT=224
INPUT_WEIGHT=224
OUTPUT_DIM=2

class AlexNet():
    def __init__(self):
        #basic environment
        #self.input_dim=INPUT_DIM
        self.output_dim=OUTPUT_DIM
    #pytorch-like
    def conv2d(self,X,out_channels, kernel_size, strides, padding,regularizer,name):
        logits_conv = tf.layers.conv2d(
            inputs=X,filters=out_channels,kernel_size=kernel_size,strides=strides,
            padding=padding,activation=tf.nn.relu,use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.initializers.constant(),
            kernel_regularizer=regularizer,bias_regularizer=regularizer,
            activity_regularizer=None,
            trainable=True,
            name=name
        )
        return logits_conv

    # pytorch-like
    def linear(self, X, units,regularizer,name):
        logits_fc = tf.layers.dense(
            inputs=X,
            units=units,
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.initializers.constant(),
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
            activity_regularizer=None,
            trainable=True,
            name=name
        )
        return logits_fc


    def forward(self,X,regularizer):
        #-------------------------------------conv1------------------------------------------------------------
        logits_conv1=self.conv2d(X,96,(11,11),(4,4),"SAME",regularizer,"logits_conv1")
        #max pooling
        logits_conv1=tf.layers.max_pooling2d(inputs=logits_conv1,pool_size=(3,3),strides=(2,2),padding="VALID")
        print("logits_conv1.shape",logits_conv1.shape)

        # -------------------------------------conv2------------------------------------------------------------
        logits_conv2 = self.conv2d(logits_conv1, 256, (5, 5), (1, 1), "SAME", regularizer, "logits_conv2")
        # max pooling
        logits_conv2 = tf.layers.max_pooling2d(inputs=logits_conv2, pool_size=(3, 3), strides=(2, 2), padding="VALID")
        print("logits_conv2.shape", logits_conv2.shape)

        # -------------------------------------conv3------------------------------------------------------------3
        logits_conv3 = self.conv2d(logits_conv2, 384, (3, 3), (1, 1), "SAME", regularizer, "logits_conv3")
        print("logits_conv3.shape", logits_conv3.shape)
        # -------------------------------------conv4------------------------------------------------------------
        logits_conv4 = self.conv2d(logits_conv3, 384, (3, 3), (1, 1), "SAME", regularizer, "logits_conv4")
        print("logits_conv4.shape", logits_conv4.shape)

        # -------------------------------------conv5------------------------------------------------------------
        logits_conv5 = self.conv2d(logits_conv4, 256, (3, 3), (1, 1), "SAME", regularizer, "logits_conv5")
        # max pooling
        logits_conv5 = tf.layers.max_pooling2d(inputs=logits_conv5, pool_size=(3, 3), strides=(2, 2), padding="VALID")
        print("logits_conv5.shape", logits_conv5.shape)

        #reshape to connect to fully conected layer
        plat_logits_conv5=tf.layers.flatten(inputs=logits_conv5)
        print("plat_logits_conv5:",plat_logits_conv5.shape)

        # -------------------------------------FC1------------------------------------------------------------
        logits_fc1=self.linear(plat_logits_conv5,4096,regularizer,"logits_fc1")
        print("logits_fc1.shape", logits_fc1.shape)
        # -------------------------------------FC2------------------------------------------------------------
        logits_fc2 = self.linear(logits_fc1, 4096, regularizer, "logits_fc2")
        print("logits_fc2.shape", logits_fc2.shape)
        # -------------------------------------FC3------------------------------------------------------------
        logits_fc3 = self.linear(logits_fc2, self.output_dim, regularizer, "logits_fc3")
        print("logits_fc3.shape", logits_fc3.shape)
        return logits_fc3


if __name__=="__main__":
    input=tf.random_normal(shape=(10,224,224,3))

    model=AlexNet()
    model.forward(X=input,regularizer=None)




'''
#--------------------------------------Old Version--------------------------------------#
class AlexNet():
    def __init__(self):
        #basic environment
        #self.input_dim=INPUT_DIM
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
            conv=tf.nn.conv2d(input=X,filter=filter_weights,strides=[1,stride[0],stride[1],1],padding=padding,data_format="NHWC")
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
        #logits_fc3 = tf.nn.softmax(logits=logits_fc3,axis=1)

        return logits_fc3


if __name__=="__main__":
    input=tf.random_normal(shape=(10,32,32,3))

    model=AlexNet()
    model.forward(X=input,regularizer=None)

'''







