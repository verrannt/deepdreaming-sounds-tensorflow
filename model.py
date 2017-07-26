'''
A simple Convolutional Neural Network with 4 convolutional and 2 feed forward
layers.

The init function will initialize the hyperparameters, create the dataflow graph
and the functions for backpropagation and performance evaluation.

Additional functions are provided in order to make the dataflow graph more
readable.

@date: 2017-06-20
'''

import tensorflow as tf
import numpy as np
from utilities import variable_summaries
from cnn_architectures import Architectures

class CNN():

    def __init__(self, input_shape, kernel_size, n_classes):
        '''Initializes the model with its hyperparameters and builds the data
        flow graph, optimization/evaluation step and summary requirements.

        @params
        input_shape: the shape of a single input image
        kernel_size: the size of the kernels of the convolutional layers
        n_classes: the number of classes, needed for the output of the network
        '''

        # Get shapes for the layers
        self.architecture = Architectures.fat_shallow(kernel_size, n_classes)

        # Init parameters
        input_height, input_width = input_shape
        self.x = tf.placeholder(tf.float32, [None, input_height, input_width])
        self.labels = tf.placeholder(tf.float32, [None, n_classes])
        self.x_image = tf.reshape(self.x, [-1, input_height, input_width, 1])
        self.keep_prob = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32)

        # Data flow graph
        self.conv1 = self.conv_layer(self.x_image, 'conv1')
        self.conv2 = self.conv_layer(self.conv1, 'conv2')
        self.pool2 = self.max_pool(self.conv2, 'pool2')
        self.pool2_dropout = tf.nn.dropout(self.pool2, self.keep_prob)
        self.conv3 = self.conv_layer(self.pool2_dropout, 'conv3')
        self.conv4 = self.conv_layer(self.conv3, 'conv4')
        self.pool4 = self.max_pool(self.conv4, 'pool4')
        self.flat = self.flatten(self.pool4)
        self.fc1 = self.activate(self.fc_layer(self.flat, 'fc1'))
        self.fc1_dropout = tf.nn.dropout(self.fc1, self.keep_prob)
        self.output = self.fc_layer(self.fc1_dropout, 'fc2')

        # Cross entropy (variable_scope for tensorboard summaries)
        with tf.variable_scope('cross_entropy'):
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.output))
        # L2 LOSS CURRENTLY NOT IN USE BECAUSE IT CAUSED WORSE PERFORMANCE
        # # Calculate L2 loss
        # all_vars = tf.trainable_variables()
        # self.l2 = 0.001 * tf.add_n([tf.nn.l2_loss(v) for v in all_vars
        #     if 'conv_kernel' in v.name or 'fc_weights' in v.name and not 'bias' in v.name])
        # # Add L2 loss to cross entropy
        # self.loss = tf.add(self.cross_entropy, self.l2, name="loss")
        # Optimizer step
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)

        # Evaluate prediction accuracy
        with tf.variable_scope('correct_prediction'):
            self.correct_prediction = tf.equal(tf.argmax(self.output,1), tf.argmax(self.labels,1))
        with tf.variable_scope('accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # Add to summary
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('cross_entropy', self.cross_entropy)
        self.merged = tf.summary.merge_all()

    def flatten(self, input):
        '''Flattens an input tensor in order to fit it from a convolutional
        into a fully connected layer

        @returns
        the flattened input tensor
        '''
        shape = input.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        return tf.reshape(input, [-1, dim])

    def activate(self, input):
        '''Performs an element-wise leaky ReLU activation on the input tensor

        @returns
        leaky ReLU activation applied to the input tensor
        '''
        leak = 0.2
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * input + f2 * abs(input)

    def max_pool(self, input, name):
        '''Performs max pooling on the input tensor

        @params
        input: the input tensor
        name: name of the pooling layer for tensorboard visualization

        @returns
        max pooling applied to the input tensor
        '''
        return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1],
            padding='SAME', name=name)

    def conv_layer(self, input, name):
        '''Creates a convolutional layer with variable scopes for tensorboard
        visualization. Shapes for the kernel/bias are provided by the
        cnn_architectures class. Does perform an activation using the
        self.activate() function.

        @params
        input: input tensor to the layer
        name: specifies whether it's a fully-connected or convolutional layer

        @returns
        a tensorflow op for the whole layer
        '''
        with tf.variable_scope(name):
            shape = self.architecture[name]
            with tf.variable_scope('conv_kernel'):
                kernel = self.get_weights('kernel', shape)
                variable_summaries(kernel)
            with tf.variable_scope('convolution'):
                conv = tf.nn.conv2d(input, kernel, strides=[1,1,1,1], padding='SAME')
                variable_summaries(conv)
            with tf.variable_scope('conv_bias'):
                bias = self.get_bias('conv_bias', shape)
                variable_summaries(bias)
            with tf.variable_scope('convolution_with_bias'):
                conv_bias = tf.nn.bias_add(conv, bias)
                tf.summary.histogram('pre_activations_conv', conv_bias)
            return self.activate(conv_bias)

    def fc_layer(self, input, name):
        '''Creates a feed-forward/fully-connected layer with variable scopes
        for tensorboard visualization. Shapes for the weights/bias are provided
        by the cnn_architectures class. Does NOT perform an activation.

        @params
        input: input tensor to the layer
        name: specifies whether it's a fully-connected or convolutional layer

        @returns
        a tensorflow op for the whole layer
        '''
        with tf.variable_scope(name):
            input_shape = input.get_shape().as_list()
            shape = [input_shape[1], self.architecture[name][1]]
            with tf.variable_scope('fc_weights'):
                weights = self.get_weights('weights', shape)
                variable_summaries(weights)
            with tf.variable_scope('fc_bias'):
                bias = self.get_bias('fc_bias', shape)
                variable_summaries(bias)
            with tf.variable_scope('fully_connected'):
                fc = tf.nn.bias_add(tf.matmul(input, weights), bias, name='fc_out')
                tf.summary.histogram('pre_activations_fc', fc)
            return fc

    def get_weights(self, name, shape):
        '''Initializes weights for fully-connected or convolutional layers,
        resp.

        @params
        name: specifies whether it's for a fully-connected or conv. layer
        shape: specifies the shape the weight matrix is supposed to have

        @returns
        a tensor of shape *shape*
        '''
        return tf.get_variable(name, shape,
            initializer = tf.random_normal_initializer(mean=0,stddev=0.8))

    def get_bias(self, name, shape):
        '''Initializes a bias for fully-connected or convolutional layers,
        resp.

        @params
        name: specifies whether it's for a fully-connected or conv. layer
        shape: specifies the shape the bias vector is supposed to have

        @returns
        a 1d tensor
        '''
        return tf.get_variable(name, shape[-1],
            initializer = tf.constant_initializer(0.1))
