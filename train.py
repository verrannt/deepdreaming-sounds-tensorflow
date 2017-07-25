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

# NOT NEEDED RIGHT NOW BECAUSE WE ARE USING WAV FILES
# Download the modified urbansound dataset if it is not already present
# dir_content = listdir("UrbanSound8K_modified")
# if not 'urbansound.pkl' in dir_content:
# 	# TODO get the right adress
# 	url = 'http://s33.filefactory.com/get/f/4o4d4li32zwl/2709b0a6c35442fe/UrbanSound8K_modified_v2.zip'
# 	print("Dataset not found. Downloading 468MB, please wait ...")
# 	urllib.request.urlretrieve(url, "./UrbanSound8K_modified/urbansound.zip")
# 	zip_ref = zipfile.ZipFile("./UrbanSound8K_modified/urbansound.zip", 'r')
# 	zip_ref.extractall("./UrbanSound8K_modified/")
# 	zip_ref.close()
# 	os.remove("./UrbanSound8K_modified/urbansound.zip")

# Get values for number of iterations, batch size and path to the dataset;
# if not given, use standard values
args = sys.argv
if len(sys.argv) == 1:
	n_iterations = 50000
	batch_size = 100
	# path = "./UrbanSound8K_modified/urbansound.pkl"
	path = "../TrainingData/UrbanSound8K_modified_v2/audio/"
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
util = Batchgeneration(path)

def train(path, n_iterations, batch_size):
	'''
	Train and test network on modified UrbanSound8k dataset
	'''
	# Base learning rate
	learning_rate = 0.01
	# Initialize the model specified in the model.py file
	model = CNN(input_shape = (129, 13), kernel_size = 3,
		n_classes = 9, learning_rate = learning_rate) # changed Transposed here
	# Initialize saver class
	saver = tf.train.Saver(tf.trainable_variables())

	with tf.Session() as ses:
		# Initialize session and variables
		ses.run(tf.global_variables_initializer())
		# Initialize writer for tensorboard summaries
		writer = tf.summary.FileWriter(logdir="logs/tensorboard", graph=ses.graph)
		# Initialize proto to save graph
		tf.train.write_graph(ses.graph_def, 'logs/model', 'graph.pb')

		for i in range(n_iterations):
			# Generate the batch with specified batch size using utilities.py's method
			# to generate the batch from wav files or the pickle file
			trainX,trainY,valX,valY,testX,testY = util.generate_batch_from_wav(batch_size)

			# Training step
			train_acc, summary, _ = ses.run(
				[model.accuracy, model.merged, model.train_step],
				feed_dict = {model.x:trainX, model.labels:trainY,
					model.keep_prob:0.5, model.learning_rate:learning_rate})
			print("Step %d -- Training accuracy: %g"%(i, train_acc))

			# Dynamic learning rate: collect last 200 training accuracies,
			# if their mean is above 70%, reduce the learning rate to 0.0001
			accuracies.append(train_acc)
			if len(accuracies) > 200:
				accuracies.pop(0)
			if np.mean(l) > 0.7:
				learning_rate = 0.0001

			# Validation step
			if i % val_step == 0:
				val_acc = ses.run(
					model.accuracy,
					feed_dict = {model.x:valX, model.labels:valY,
						model.keep_prob:1.0, model.learning_rate:learning_rate})
				print("Step %d -- Validate accuracy: %g"%(i, val_acc))

			# Save session every 500 steps
			if i % 500 == 0:
				saver.save(ses, "logs/model/saver/model-{}".format(i))

			writer.add_summary(summary, i)

		# Final testing evaluation
		test_acc = ses.run(
			model.accuracy,
			feed_dict={model.x:testX, model.labels:testY, model.keep_prob:1.0})
		print("Test accuracy: %g"%(test_acc))

		# Close the summary writer
		writer.close()

train(path, n_iterations, batch_size)

# def train_mnist(path, n_iterations, batch_size):
# 	''' Train and test network on MNIST dataset '''
# 	from tensorflow.examples.tutorials.mnist import input_data
# 	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 	model = CNN(input_shape = (28, 28), kernel_size = 5, n_classes = 10)
# 	init = tf.global_variables_initializer()
# 	with tf.Session() as ses:
# 		writer = tf.summary.FileWriter(logdir="./logs/tensorboard/mnist",graph_def=model)
# 		ses.run(init)
# 		for i in range(n_iterations):
# 			trainBatch = mnist.train.next_batch(batch_size)
# 			trainX, trainY = trainBatch[0], trainBatch[1]
# 			train_acc, summary, _ = ses.run(
# 				[model.accuracy, model.merged, model.train_step],
# 				feed_dict = {model.x:trainX, model.labels:trainY, model.keep_prob:0.5})
# 			print("Step %d -- Training accuracy: %g"%(i, train_acc))
# 			writer.add_summary(summary, i)
# 		test_accuracy = ses.run(
# 			model.accuracy,
# 			feed_dict={model.x:mnist.test.images, model.labels:mnist.test.labels, model.keep_prob:1.0})
# 		print("Test accuracy: %g"%(test_accuracy))
# 		writer.close()
