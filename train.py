import tensorflow as tf
import scipy.io.wavfile
import numpy as np
import matplotlib.mlab
import urllib.request
import zipfile
from os import listdir
from os.path import isfile, join
import sys
from utilities import Util
from model import CNN

# Download the modified urbansound dataset if it is not already present
dir_content = listdir()
if not 'urbansound.pkl' in dir_content:
	url = 'http://www.blog.pythonlibrary.org/wp-content/uploads/2012/06/wxDbViewer.zip'#'http://s33.filefactory.com/get/f/4o4d4li32zwl/2709b0a6c35442fe/UrbanSound8K_modified_v2.zip'
	print("Dataset not found. Downloading 468MB, please wait ...")
	urllib.request.urlretrieve(url, "./UrbanSound8K_modified/urbansound.zip")
	zip_ref = zipfile.ZipFile("./UrbanSound8K_modified/urbansound.zip", 'r')
	zip_ref.extractall("./UrbanSound8K_modified/")
	zip_ref.close()
	os.remove("./UrbanSound8K_modified/urbansound.zip")

# Get values for number of iterations, batch size and path to the dataset;
# if not given, use standard values
args = sys.argv
if len(sys.argv) == 1:
	n_iterations = 5000
	batch_size = 100
	path = "./UrbanSound8K_modified/urbansound.pkl"
elif len(sys.argv) == 2:
	n_iterations = int(args[1])
	batch_size = 100
	path = "./UrbanSound8K_modified/urbansound.pkl"
elif len(sys.argv) == 3:
	n_iterations = int(args[1])
	batch_size = int(args[2])
	path = "./UrbanSound8K_modified/urbansound.pkl"
else:
	n_iterations = int(args[1])
	batch_size = int(args[2])
	path = str(args[3])

# Specify after how many steps we want to make a validation step
val_step = 10
# Initialize the utilities class for generating batches
util = Util(path)

def train_mnist(path, n_iterations, batch_size):
	''' Train and test network on MNIST dataset '''
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	model = CNN(input_shape = (28, 28), kernel_size = 5, n_classes = 10)
	init = tf.global_variables_initializer()
	with tf.Session() as ses:
		writer = tf.summary.FileWriter(logdir="./logs/tensorboard/mnist",graph_def=model)
		ses.run(init)
		for i in range(n_iterations):
			trainBatch = mnist.train.next_batch(batch_size)
			trainX, trainY = trainBatch[0], trainBatch[1]
			train_acc, summary, _ = ses.run(
				[model.accuracy, model.merged, model.train_step],
				feed_dict = {model.x:trainX, model.labels:trainY, model.keep_prob:0.5})
			print("Step %d -- Training accuracy: %g"%(i, train_acc))
			writer.add_summary(summary, i)
		test_accuracy = ses.run(
			model.accuracy,
			feed_dict={model.x:mnist.test.images, model.labels:mnist.test.labels, model.keep_prob:1.0})
		print("Test accuracy: %g"%(test_accuracy))
		writer.close()


def train_urbansound(path, n_iterations, batch_size):
	''' Train and test network on UrbanSound8k dataset '''

	# Initialize the model specified in the model.py file
	model = CNN(input_shape = (129, 13), kernel_size = 3, n_classes = 9) # changed Transposed here
	# Initialize saver class
	saver = tf.train.Saver(tf.trainable_variables())

	with tf.Session() as ses:
		# Initialize session and variables
		ses.run(tf.global_variables_initializer())
		# Initialize writer for tensorboard summaries
		writer = tf.summary.FileWriter(logdir="logs/tensorboard", graph=ses.graph)
		# Initialize proto to save graph
		tf.train.write_graph(ses.graph_def, 'logs/model','graph.pb')

		for i in range(n_iterations):
			# Generate the batch with specified batch size using utilities.py's method
			trainX,trainY,valX,valY,testX,testY = util.generate_batch_from_pickle(batch_size)

			# Training step
			train_acc, summary, _ = ses.run(
				[model.accuracy, model.merged, model.train_step],
				feed_dict = {model.x:trainX, model.labels:trainY, model.keep_prob:0.7})
			print("Step %d -- Training accuracy: %g"%(i, train_acc))

			writer.add_summary(summary, i)

			# Validation step
			if i % val_step == 0:
				val_acc = ses.run(
					model.accuracy,
					feed_dict = {model.x:valX, model.labels:valY, model.keep_prob:1.0})
				print("Step %d -- Validate accuracy: %g"%(i, val_acc))

		# Final testing evaluation
		test_acc = ses.run(
			model.accuracy,
			feed_dict={model.x:testX, model.labels:testY, model.keep_prob:1.0})
		print("Test accuracy: %g"%(test_acc))

		# Save the weights
		saver.save(ses, "logs/model/model")
		# Close the summary writer
		writer.close()

# train_mnist(path, n_iterations, batch_size)
train_urbansound(path, n_iterations, batch_size)
