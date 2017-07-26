'''
Evaluate the network on a specified number of testing samples from the modified
UrbanSound8K dataset and print out the results.

@date: 2017-07-26
'''

import random
import tensorflow as tf
from os import listdir
from scipy.io import wavfile
from utilities import Batchgeneration
from model import CNN

path = "./UrbanSound8K_modified/"

# Initialize the utilities class for generating batches
util = Batchgeneration(path)
# and the outputs list to hold the network's outputs.
outputs = []
# Initialize the model
model = CNN(input_shape = (129, 13), kernel_size = 3, n_classes = 9) # changed Transposed here
# Initialize the saver
# saver = tf.train.Saver()
with tf.Session() as ses:
    # Initialize the saver from meta graph
    saver = tf.train.import_meta_graph("logs/model/model-10500.meta")
    # and restore session from disk.
    saver.restore(ses, "logs/model/model-10500")

    # Generate one testing batch that contains 9*33 testing samples in testX and
    # their corresponding labels in testY.
    _,_,_,_, testX, testY = util.generate_batch_from_pickle(1000)
    # For each of the samples
    for i in range(len(testX)):
        # get the network's prediction in what class it belongs
        output = ses.run(model.output, feed_dict = {
            model.x: testX, model.labels: testY,
            model.keep_prob: 1.0, model.learning_rate: 0.001})
        # and append it to the outputs array.
        outputs.append(output)

assert len(outputs) == len(testY), "WARNING: Length of output list and label \n\
list do not match!"
# Print the network outputs in comparison to the labels
print("= OUTPUTS OF NETWORK VS ORIGINAL LABELS =\n\
========================================")
for i in range(len(outputs)):
    print(outputs[i] + " ... " + testY[i])

###########

# classes = ['air_conditioner', 'car_horn', 'children_playing',
#     'dog_bark', 'drilling', 'engine_idling', 'jackhammer', 'siren',
#     'street_music']
# samples = []
#
# for sample in n_samples:
#     # choose a random class
#     random_class = random.choice(classes)
#     # get names of all TEST files in class
#     all_filenames = listdir(path + random_class)
#     test_filenames = [f for f in all_files if f.split("-")[0] == "TEST"]
#     # choose random test file
#     random_test_filename = random.choice(test_filenames)
#     # get wav file
#     wav_file = wavfile.read(path + class_name +"/"+ random_test_filename))[1]
#     # convert to spectogram
#     spec = Batchgeneration.spectogram(wav_file)
