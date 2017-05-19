'''
A simple Convolutional Neural Network with 5 convolutional and 2 feed forward
layers.

@date: 2017-05-19
'''

import tensorflow as tf
import numpy as np
from cnn_architectures import Architectures

class CNN():
    def __init__(self, kernel_size, classes):
        # Get the shapes of simplified VGG16 architecture
        architecture = Architectures.vgg16_downsized(kernel_size=5,
                                                     amount_classes=1000)
        # Init parameters
        self.x = tf.placeholder(tf.float32, [None, 1024])
        self.yHat = tf.placeholder(tf.float32, [None, classes])
        self.x_image = tf.reshape(self.x, [-1,32,32,1])
        self.keep_prob = tf.placeholder(tf.float32)
        
        # Data flow graph
        self.conv1_1 = self.conv_layer(self.x_image, "conv1_1")
        self.pool1 = self.max_pool(self.conv1_1, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.pool2 = self.max_pool(self.conv2_1, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.pool3 = self.max_pool(self.conv3_1, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.pool4 = self.max_pool(self.conv4_1, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.pool5 = self.max_pool(self.conv5_1, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.act6 = activate(self.fc6)
        self.act6_dropout = tf.nn.dropout(self.act6, self.keep_prob)

        self.fc7 = self.fc_layer(self.act6_dropout, "fc7")

        self.y = tf.nn.softmax(self.fc7, name="network_output")

    def activate(input):
        return tf.nn.relu(input)

    def max_pool(self, input, name):
        return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1],
            padding='SAME', name=name)

    def conv_layer(self, input, name):
        with tf.variable_scope(name):
            shape = architecture[name]
            kernel = self.get_kernel(shape)
            conv = tf.nn.conv2d(input, kernel, strides=[1,1,1,1], padding='SAME')
            bias = self.get_bias(shape)
            conv_bias = tf.nn.bias_add(conv, bias)
            return activate(conv_bias)

    def fc_layer(self, input, name):
        with tf.variable_scope(name):
            shape = input.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(input, [-1, dim])
            weights = self.get_weights(shape)
            bias = self.get_bias(shape)
            fc = tf.nn.bias_add(tf.matmul(x, weights), bias)
            return fc

    def get_kernel(self, shape):
        return tf.get_variable("kernel", shape,
            initializer = tf.random_normal_initializer(mean=0,stddev=0.1))

    def get_weights(self, shape):
        return tf.get_variable("weights", shape,
            initializer = tf.random_normal_initializer(mean=0,stddev=0.1))

    def get_bias(self, shape):
        return tf.get_variable("bias", shape[:-1],
            initializer = tf.constant_initializer(0.0))
