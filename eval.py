'''
Evaluate the network on a specified number of testing samples from the modified
UrbanSound8K dataset and print out the results.

@date: 2017-07-26
'''

import random
import numpy as np
import tensorflow as tf
from os import listdir
from scipy.io import wavfile
from utilities import Batchgeneration
from model import CNN

# Initialize the utilities class for generating batches
util = Batchgeneration("./UrbanSound8K_modified/")

with tf.gfile.GFile('./logs/model/output.pb', "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(
        graph_def,
        input_map=None,
        return_elements=None,
        name="prefix",
        op_dict=None,
        producer_op_list=None)

# for op in graph.get_operations():
#         print(op.name)

x = graph.get_tensor_by_name('prefix/input:0')
keep_prob = graph.get_tensor_by_name('prefix/keep_prob:0')
y = graph.get_tensor_by_name('prefix/fc2/fully_connected/fc_out:0')

with tf.Session(graph = graph) as ses:
    _,_,_,_, testX, testY = util.generate_batch_from_pickle(100)
    output = ses.run(y, feed_dict={x: testX, keep_prob: 1.0})
    print("Shape of input: " + str(np.shape(testX)))
    print("Type of input: " + str(type(testX)))
    print("Shape of output: " + str(len(output)))
    print("Type of output: " + str(type(output)))
    print("Shape of labels: " + str(len(testY)))
    print("Type of labels: " + str(type(testY)))
    print(output[0])
    print(sum(output[0]))
    print(testY[0])

assert len(output) == len(testY), "WARNING: Length of output list and label \
list do not match!"
# Print the network outputs in comparison to the labels
print("= OUTPUTS OF NETWORK VS ORIGINAL LABELS =\n\
========================================")
for i in range(len(output)):
    print(outputs[i] + " ... " + testY[i])
