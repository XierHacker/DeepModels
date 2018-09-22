import time
import numpy as np
import tensorflow as tf

#you can modify this according your task
INPUT_HIGHT=224
INPUT_WEIGHT=224
OUTPUT_DIM=2


def batch_norm(inputs, training):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return tf.layers.batch_normalization(
      inputs=inputs,
      axis=3,
      center=True,
      scale=True,
      training=training,
      fused=True)



def fixed_padding(inputs, kernel_size):
    """Pads the input along the spatial dimensions independently of input size.
    Args:
    inputs: A tensor of size [batch, height_in, width_in, channels]
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.Should be a positive integer.
    Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],[pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size)
    conv=tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer()
    )
    return conv

def max_pooling_fixed_padding(inputs,kernel_size,strides):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size)
    max_pool=tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=kernel_size,
        strides=strides
    )
    return max_pool


def building_block_v1(inputs, filters, strides,training):
  """A single block for ResNet v1, without a bottleneck.

  Convolution then batch normalization then ReLU as described by:
    Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs
  inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides)
  inputs = batch_norm(inputs, training)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=1)
  inputs = batch_norm(inputs, training)

  inputs += shortcut
  inputs = tf.nn.relu(inputs)
  return inputs


def _building_block_v2(inputs, filters, training, projection_shortcut):
  """A single block for ResNet v2, without a bottleneck.

  Batch normalization then ReLu then convolution as described by:
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/pdf/1603.05027.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs
  inputs = batch_norm(inputs, training)
  inputs = tf.nn.relu(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides)

  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=1)

  return inputs + shortcut


def _bottleneck_block_v1(inputs, filters, training, projection_shortcut,strides):
  """A single block for ResNet v1, with a bottleneck.

  Similar to _building_block_v1(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)
    shortcut = batch_norm(inputs=shortcut, training=training)

  inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1, strides=1)
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides)
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(inputs=inputs, filters=4 * filters, kernel_size=1, strides=1)
  inputs = batch_norm(inputs, training, data_format)
  inputs += shortcut
  inputs = tf.nn.relu(inputs)

  return inputs


def _bottleneck_block_v2(inputs, filters, training, projection_shortcut,strides):
  """A single block for ResNet v2, with a bottleneck.

  Similar to _building_block_v2(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

  Adapted to the ordering conventions of:
    Batch normalization then ReLu then convolution as described by:
      Identity Mappings in Deep Residual Networks
      https://arxiv.org/pdf/1603.05027.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs
  inputs = batch_norm(inputs, training)
  inputs = tf.nn.relu(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1, strides=1)

  inputs = batch_norm(inputs, training)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides)

  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(inputs=inputs, filters=4 * filters, kernel_size=1, strides=1)

  return inputs + shortcut



def block_layer(inputs, filters, bottleneck, block_fn, blocks,training, name):
    """Creates one layer of blocks for the ResNet model.
    Args:
    inputs: A tensor of size [batch, height_in, width_in, channels]
    filters: The number of filters for the first convolution of the layer.
    bottleneck: Is the block created a bottleneck block.
    block_fn: The block to use within the model, either `building_block` or `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If greater than 1, this layer will ultimately downsample the input.
    training: Either True or False, whether we are currently training the model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.

    Returns:
    The output tensor of the block layer.
    """

    # Bottleneck blocks end with 4x the number of filters as they start with
    #filters_out = filters * 4 if bottleneck else filters

    #def projection_shortcut(inputs):
    #  return conv2d_fixed_padding(inputs=inputs, filters=filters_out, kernel_size=1, strides=strides)

    # Only the first block per block_layer uses projection_shortcut and strides
    #inputs = block_fn(inputs, filters, training, projection_shortcut, strides)

    for i in range(0, blocks):
        #print("i:",i)
        inputs = block_fn(inputs, filters,1, training)
    return tf.identity(inputs, name)





class ResNet_18():
    def __init__(self):
        self.output_dim=OUTPUT_DIM

    def forward(self,X,training):
        #---------------------------------conv1---------------------------------------------
        logits_conv1=conv2d_fixed_padding(inputs=X,filters=64,kernel_size=7,strides=2)
        print("logits_conv1.shape",logits_conv1.shape)
        #max pooling
        logits_conv1=max_pooling_fixed_padding(inputs=logits_conv1,kernel_size=3,strides=2)
        print("logits_conv1.shape", logits_conv1.shape)

        #--------------------------------block_modual_1-------------------------------------
        #projection,channels to 64
        logits_block_modual_1=conv2d_fixed_padding(inputs=logits_conv1,filters=64,kernel_size=3,strides=1)
        print("logits_block_modual_1.shape", logits_block_modual_1.shape)
        #block modual-1(contains 2 layer)
        logits_block_modual_1=block_layer(
            inputs=logits_block_modual_1,
            filters=64,
            bottleneck=False,
            block_fn=building_block_v1,
            blocks=2,
            training=training,
            name="block_modual_1"
        )
        print("locits_block_modual_1.shape", logits_block_modual_1.shape)

        # --------------------------------block_modual_2-------------------------------------
        # projection,channels to 128
        logits_block_modual_2 = conv2d_fixed_padding(inputs=logits_block_modual_1, filters=128, kernel_size=3, strides=2)
        print("logits_block_modual_2.shape", logits_block_modual_2.shape)
        # block modual-1(contains 2 layer)
        logits_block_modual_2 = block_layer(
            inputs=logits_block_modual_2,
            filters=128,
            bottleneck=False,
            block_fn=building_block_v1,
            blocks=2,
            training=training,
            name="block_modual_2"
        )
        print("locits_block_modual_2.shape", logits_block_modual_2.shape)

        # --------------------------------block_modual_3-------------------------------------
        # projection,channels to 128
        logits_block_modual_3 = conv2d_fixed_padding(inputs=logits_block_modual_2, filters=256, kernel_size=3,strides=2)
        print("logits_block_modual_3.shape", logits_block_modual_3.shape)
        # block modual-1(contains 2 layer)
        logits_block_modual_3 = block_layer(
            inputs=logits_block_modual_3,
            filters=256,
            bottleneck=False,
            block_fn=building_block_v1,
            blocks=2,
            training=training,
            name="block_modual_3"
        )
        print("locits_block_modual_3.shape", logits_block_modual_3.shape)

        # --------------------------------block_modual_4-------------------------------------
        # projection,channels to 128
        logits_block_modual_4 = conv2d_fixed_padding(inputs=logits_block_modual_3, filters=512, kernel_size=3,strides=2)
        print("logits_block_modual_4.shape", logits_block_modual_4.shape)
        # block modual-1(contains 2 layer)
        logits_block_modual_4 = block_layer(
            inputs=logits_block_modual_4,
            filters=512,
            bottleneck=False,
            block_fn=building_block_v1,
            blocks=2,
            training=training,
            name="block_modual_4"
        )
        print("locits_block_modual_4.shape", logits_block_modual_4.shape)

        #max_pool
        logits_block_modual_4=max_pooling_fixed_padding(inputs=logits_block_modual_4,kernel_size=3,strides=2)
        print("locits_block_modual_4.shape", logits_block_modual_4.shape)

        # --------------------------------fully connected layer-------------------------------------
        flatten=tf.layers.flatten(inputs=logits_block_modual_4)
        print("flatten.shape", flatten.shape)
        logits=tf.layers.dense(inputs=flatten,units=self.output_dim,activation=None,name="logits")
        print("logits.shape", logits.shape)
        return logits





class ResNet_34():
    pass

class ResNet_50():
    pass

class ResNet_101():
    pass

class ResNet_152():
    pass






'''
class AlexNet():
    def __init__(self):
        #basic environment
        #self.input_dim=INPUT_DIM
        self.output_dim=OUTPUT_DIM

    def batch_norm(self,X,training,name,reuse):
        normed=tf.layers.batch_normalization(
            inputs=X,
            axis=3,
            training=training,
            fused=True,
            name=name,
            reuse=reuse
        )
        return normed

    #pytorch-like
    def conv2d(self,X,out_channels, kernel_size, strides, padding,regularizer,name,reuse):
        logits_conv = tf.layers.conv2d(
            inputs=X,filters=out_channels,kernel_size=kernel_size,strides=strides,
            padding=padding,activation=tf.nn.relu,use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.initializers.constant(),
            kernel_regularizer=regularizer,bias_regularizer=regularizer,
            activity_regularizer=None, trainable=True,name=name,reuse=reuse
        )
        return logits_conv

    # pytorch-like
    def linear(self, X, units,activation,regularizer,keep_rate,name,reuse):
        logits_fc = tf.layers.dense(
            inputs=X,units=units,activation=activation,use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.initializers.constant(),
            kernel_regularizer=regularizer,bias_regularizer=regularizer,activity_regularizer=None,
            trainable=True,name=name,reuse=reuse
        )
        #dropout
        logits_fc=tf.layers.dropout(inputs=logits_fc,rate=1-keep_rate)
        return logits_fc


    def forward(self,X,regularizer,keep_rate,training,reuse):
        #-------------------------------------conv1------------------------------------------------------------
        logits_conv1=self.conv2d(X,96,(11,11),(4,4),"SAME",regularizer,"logits_conv1",reuse)
        #batch_norm
        logits_conv1=self.batch_norm(logits_conv1,training,"conv1_normed",reuse)
        #max pooling
        logits_conv1=tf.layers.max_pooling2d(inputs=logits_conv1,pool_size=(3,3),strides=(2,2),padding="VALID")
        #print("logits_conv1.shape",logits_conv1.shape)

        # -------------------------------------conv2------------------------------------------------------------
        logits_conv2 = self.conv2d(logits_conv1, 256, (5, 5), (1, 1), "SAME", regularizer, "logits_conv2",reuse)
        # batch_norm
        logits_conv2 = self.batch_norm(logits_conv2,training,"conv2_normed",reuse)
        # max pooling
        logits_conv2 = tf.layers.max_pooling2d(inputs=logits_conv2, pool_size=(3, 3), strides=(2, 2), padding="VALID")
        #print("logits_conv2.shape", logits_conv2.shape)

        # -------------------------------------conv3------------------------------------------------------------3
        logits_conv3 = self.conv2d(logits_conv2, 384, (3, 3), (1, 1), "SAME", regularizer, "logits_conv3",reuse)
        # batch_norm
        logits_conv3 = self.batch_norm(logits_conv3,training,"conv3_normed",reuse)
        #print("logits_conv3.shape", logits_conv3.shape)
        # -------------------------------------conv4------------------------------------------------------------
        logits_conv4 = self.conv2d(logits_conv3, 384, (3, 3), (1, 1), "SAME", regularizer, "logits_conv4",reuse)
        # batch_norm
        logits_conv4 = self.batch_norm(logits_conv4,training,"conv4_normed",reuse)
        #print("logits_conv4.shape", logits_conv4.shape)

        # -------------------------------------conv5------------------------------------------------------------
        logits_conv5 = self.conv2d(logits_conv4, 256, (3, 3), (1, 1), "SAME", regularizer, "logits_conv5",reuse)
        # batch_norm
        logits_conv5 = self.batch_norm(logits_conv5,training,"conv5_normed",reuse)
        # max pooling
        logits_conv5 = tf.layers.max_pooling2d(inputs=logits_conv5, pool_size=(3, 3), strides=(2, 2), padding="VALID")
        #print("logits_conv5.shape", logits_conv5.shape)

        #reshape to connect to fully conected layer
        plat_logits_conv5=tf.layers.flatten(inputs=logits_conv5)
        #print("plat_logits_conv5:",plat_logits_conv5.shape)

        # -------------------------------------FC1------------------------------------------------------------
        logits_fc1=self.linear(plat_logits_conv5,4096,tf.nn.relu,regularizer,keep_rate,"logits_fc1",reuse)
        #print("logits_fc1.shape", logits_fc1.shape)
        # -------------------------------------FC2------------------------------------------------------------
        logits_fc2 = self.linear(logits_fc1, 4096,tf.nn.relu, regularizer,keep_rate, "logits_fc2",reuse)
        #print("logits_fc2.shape", logits_fc2.shape)
        # -------------------------------------FC3------------------------------------------------------------
        logits_fc3 = self.linear(logits_fc2, self.output_dim, None,regularizer, keep_rate,"logits_fc3",reuse)
        #print("logits_fc3.shape", logits_fc3.shape)
        return logits_fc3

'''





if __name__=="__main__":
    input=tf.random_normal(shape=(10,224,224,3))

    model=ResNet_18()
    model.forward(X=input,training=True)