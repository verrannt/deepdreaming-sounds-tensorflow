'''
A simple Convolutional Neural Network with 5 convolutional and 2 feed forward
layers.

@date: 2017-06-20
'''

import tensorflow as tf
import numpy as np
from utilities import variable_summaries
from cnn_architectures import Architectures

class CNN():

    def __init__(self, input_shape, kernel_size, n_classes, learning_rate):
        # Get shapes for the layers
        self.architecture = Architectures.fat_shallow(kernel_size, n_classes)
        self.learning_rate = learning_rate

        # Init parameters
        input_height, input_width = input_shape
        self.x = tf.placeholder(tf.float32, [None, input_height, input_width])
        self.labels = tf.placeholder(tf.float32, [None, n_classes])
        self.x_image = tf.reshape(self.x, [-1, input_height, input_width, 1])
        self.keep_prob = tf.placeholder(tf.float32)

        # Data flow graph
        self.conv1 = self.conv_layer(self.x_image, "conv1")
        self.conv2 = self.conv_layer(self.conv1, "conv2")
        self.pool2 = self.max_pool(self.conv2, "pool2")
        self.pool2_dropout = tf.nn.dropout(self.pool2, self.keep_prob)

        self.conv3 = self.conv_layer(self.pool2_dropout, "conv3")
        self.conv4 = self.conv_layer(self.conv3, "conv4")
        self.pool4 = self.max_pool(self.conv4, "pool4")

        # self.conv5 = self.conv_layer(self.pool4, "conv5")
        # self.conv5_1 = self.conv_layer(self.conv5, "conv5_1")
        # self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")

        self.flat = self.flatten(self.pool4)

        self.fc1 = self.activate(self.fc_layer(self.flat, "fc1"))
        self.fc1_dropout = tf.nn.dropout(self.fc1, self.keep_prob)
        self.fc2 = self.fc_layer(self.fc1_dropout, "fc2")
        # self.output = tf.nn.softmax(self.fc7, name="network_output")
        self.output = self.fc2

        with tf.name_scope('cross_entropy'):
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.output))
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)
        with tf.name_scope('correct_prediction'):
            self.correct_prediction = tf.equal(tf.argmax(self.output,1), tf.argmax(self.labels,1))
        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('cross_entropy', self.cross_entropy)
        self.merged = tf.summary.merge_all()

    def flatten(self, input):
        shape = input.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        return tf.reshape(input, [-1, dim])

    def activate(self, input):
        # NORMAL ReLU
        # return tf.nn.relu(input)
        # LEAKY ReLU
        leak = 0.2
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * input + f2 * abs(input)

    def max_pool(self, input, name):
        return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1],
            padding='SAME', name=name)

    def conv_layer(self, input, name):
        with tf.variable_scope(name):
            shape = self.architecture[name]
            print(shape)
            with tf.name_scope('conv_kernel'):
                kernel = self.get_weights("kernel", shape)
                variable_summaries(kernel)
            with tf.name_scope('convolution'):
                conv = tf.nn.conv2d(input, kernel, strides=[1,1,1,1], padding='SAME')
                variable_summaries(conv)
            with tf.name_scope('conv_bias'):
                bias = self.get_conv_bias(shape)
                variable_summaries(bias)
            with tf.name_scope('convolution_with_bias'):
                conv_bias = tf.nn.bias_add(conv, bias)
                tf.summary.histogram('pre_activations_conv', conv_bias)
            return self.activate(conv_bias)

    def fc_layer(self, input, name):
        with tf.variable_scope(name):
            input_shape = input.get_shape().as_list()
            shape = [input_shape[1], self.architecture[name][1]]
            with tf.name_scope('fc_weights'):
                weights = self.get_weights("weights", shape)
                variable_summaries(weights)
            with tf.name_scope('fc_bias'):
                bias = self.get_fc_bias(shape)
                variable_summaries(bias)
            with tf.name_scope('fully_connected'):
                fc = tf.nn.bias_add(tf.matmul(input, weights), bias)
                tf.summary.histogram('pre_activations_fc', fc)
            return fc

    def get_weights(self, name, shape):
        # original stddev 0.1
        return tf.get_variable(name, shape,
            initializer = tf.random_normal_initializer(mean=0,stddev=0.8))

    def get_conv_bias(self, shape):
        return tf.get_variable("conv_bias", shape[-1],
            initializer = tf.constant_initializer(0.1))

    def get_fc_bias(self, shape):
        return tf.get_variable("fc_bias", shape[-1],
            initializer = tf.constant_initializer(0.1))
