import tensorflow as tf
import scipy.io.wavfile
import numpy as np
import matplotlib.mlab
from os import listdir
from os.path import isfile, join
import sys

from utilities import Util
from model import CNN

arguments = sys.argv
# path = str(arguments[1])
# iterations = int(arguments[2])
# batch_size = int(arguments[3])
path = "../TrainingData/UrbanSound8K_modified_v2/audio/"
iterations = 10
batch_size = 100

val_check_freq = 10

train_accuracies = np.ones(iterations)
val_accuracies = np.ones(iterations)

train_entropies = np.zeros(iterations)
val_entropies = np.zeros(iterations)

util = Util(path)

def train(path, iterations, batch_size):
	model = CNN(input_shape = (13,129), kernel_size = 5, n_classes = 9)
	init = tf.global_variables_initializer()
	#saver = tf.train.Saver()
	with tf.Session() as ses:
		ses.run(init)
		for i in range(iterations):
			trainX,trainY,valX,valY,testX,testY = util.generate_batch(batch_size)
			train_accuracies[i], train_entropies[i], _ = ses.run(
				[model.accuracy, model.cross_entropy, model.train_step],
				feed_dict = {model.x:trainX, model.yHat:trainY, model.keep_prob:0.5})
			print("Step %d -- Training accuracy: %g"%(i, train_accuracies[i]))
			merge = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
			writer = tf.summary.FileWriter(logdir="./tensorboard/")
			if i % val_check_freq == 0:
				val_accuracies[i], val_entropies[i] = ses.run(
					[model.accuracy, model.cross_entropy],
					feed_dict = {model.x:valX, model.yHat:valY, model.keep_prob:1.0})
				print("Step %d -- Validation accuracy: %g"%(i, val_accuracies[i]))
		#saver.save(ses, "./tmp/model.ckpt")
		test_accuracy = ses.run(
			model.accuracy,
			feed_dict={model.x:testX, model.yHat:testY, model.keep_prob:1.0})
		print("Test accuracy: %g"%(test_accuracy))

train(path, iterations, batch_size)
