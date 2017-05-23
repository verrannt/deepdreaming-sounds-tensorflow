'''
A simple Convolutional Neural Network with 5 convolutional and 2 feed forward
layers.

@date: 2017-05-19
'''

import tensorflow as tf
import numpy as np
from cnn_architectures import Architectures

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

class CNN():
    def __init__(self, input_shape, kernel_size, n_classes):
        # Get the shapes of simplified VGG16 architecture
        self.architecture = Architectures.vgg16_downsized(kernel_size, n_classes)

        # Init parameters
        input_height, input_width = input_shape
        self.x = tf.placeholder(tf.float32, [None, input_height, input_width])
        self.yHat = tf.placeholder(tf.float32, [None, n_classes])
        self.x_image = tf.reshape(self.x, [-1,input_height,input_width,1])
        self.keep_prob = tf.placeholder(tf.float32)

        # Data flow graph
        self.conv1_1 = self.conv_layer(self.x_image, "conv1_1")
        self.conv2_1 = self.conv_layer(self.conv1_1, "conv2_1")
        self.pool2 = self.max_pool(self.conv2_1, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv4_1 = self.conv_layer(self.conv3_1, "conv4_1")
        self.conv5_1 = self.conv_layer(self.conv4_1, "conv5_1")
        self.pool5 = self.max_pool(self.conv5_1, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.act6 = self.activate(self.fc6)
        self.act6_dropout = tf.nn.dropout(self.act6, self.keep_prob)

        self.fc7 = self.fc_layer(self.act6_dropout, "fc7")

        self.y = tf.nn.softmax(self.fc7, name="network_output")

        self.cross_entropy = -tf.reduce_sum(self.yHat*tf.log(tf.clip_by_value(self.y,1e-10,1.0)))
        self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.yHat,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

    def activate(self, input):
        return tf.nn.relu(input)

    def max_pool(self, input, name):
        return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1],
            padding='SAME', name=name)

    def conv_layer(self, input, name):
        with tf.variable_scope(name):
            shape = self.architecture[name]
            with tf.namescope('conv_kernel'):
                kernel = self.get_kernel(shape)
                variable_summaries(kernel)
            with tf.namescope('convolution')
                conv = tf.nn.conv2d(input, kernel, strides=[1,1,1,1], padding='SAME')
                variable_summaries(conv)
            with tf.namescope('conv_bias'):
                bias = self.get_bias(shape)
                variable_summaries(bias)
            with tf.namescope('convolution+bias'):
                conv_bias = tf.nn.bias_add(conv, bias)
                variable_summaries(conv_bias)
            return self.activate(conv_bias)

    def fc_layer(self, input, name):
        with tf.variable_scope(name):
            shape = input.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(input, [-1, dim])
            shape = [dim, self.architecture[name][1]]
            with tf.namescope('fc_weights'):
                weights = self.get_weights(shape)
                variable_summaries(weights)
            with tf.namescope('fc_bias'):
                bias = self.get_bias(shape)
                variable_summaries(bias)
            with tf.name_scope('fully_connected'):
                fc = tf.nn.bias_add(tf.matmul(x, weights), bias)
                variable_summaries(fc)
            return fc

    def get_kernel(self, shape):
        return tf.get_variable("kernel", shape,
            initializer = tf.random_normal_initializer(mean=0,stddev=0.1))

    def get_weights(self, shape):
        return tf.get_variable("weights", shape,
            initializer = tf.random_normal_initializer(mean=0,stddev=0.1))

    def get_bias(self, shape):
        return tf.get_variable("bias", shape[-1],
            initializer = tf.constant_initializer(0.0))
