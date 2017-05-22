import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.mlab
from os import listdir
from os.path import isfile, join, isdir
from scipy.io import wavfile

class Util():

    def __init__(self, path):
        self.path = path
        self.classes = [f for f in listdir(self.path) if isdir(join(self.path,f))]
        self.classes = sorted(self.classes)
        self.threshold_factor = 50

    def generate_batch(self, batch_size):

        # SAMPLE LENGTH IS CURRENTLY FIXATED TO 16,000 DATAPOINTS

        samples_per_class = batch_size // 9
        assert samples_per_class <= 120, "Too many samples per class, max 120. \
            Reduce batch size!"

        trainX, valX, testX = [[] for _ in range(3)]
        train_size = samples_per_class
        val_size  = int(0.3*train_size) # [int(samples_per_class*i) for i in [0.6, 0.2, 0.2]]
        test_size = int(0.3*train_size)

        for self.class_name in self.classes:
            for _ in range(train_size):
                s = self.get_random_spec("TRAIN")
                trainX.append([self.class_name, s])
            for _ in range(val_size):
                s = self.get_random_spec("VALIDATE")
                valX.append([self.class_name, s])
            for _ in range(test_size):
                s = self.get_random_spec("TEST")
                testX.append([self.class_name, s])

        np.random.shuffle(trainX)
        np.random.shuffle(valX)
        np.random.shuffle(testX)

        trainY = np.array(self.make_one_hot(trainX))
        valY = np.array(self.make_one_hot(valX))
        testY = np.array(self.make_one_hot(testX))

        trainX = np.array(self.remove_label(trainX))
        valX = np.array(self.remove_label(valX))
        testX = np.array(self.remove_label(testX))

        return trainX, trainY, valX, valY, testX, testY

    def get_random_spec(self, set):
        all_files = listdir(join(self.path, self.class_name))
        suitable_files = [f for f in all_files if f.split("-")[0] == set]
        random_file = random.choice(suitable_files)
        wav_file = wavfile.read(join(self.path + self.class_name, random_file))[1]
        wav_file = wav_file[8000:24000]
        wav_file = wav_file/np.max(np.abs(wav_file),axis=0)
        spec = matplotlib.mlab.specgram(wav_file)[0]
        spec = self.drop_timesteps(spec)
        spec = self.sparse_sample(spec)
        return spec

    def sparse_sample(self, spec):
        for freq in spec:
            threshold = np.max(np.abs(freq))/self.threshold_factor
            for i in range(len(freq)):
                if freq[i] < threshold:
                    freq[i] = 0
        return spec

    def drop_timesteps(self, spec):
        spec_drop = []
        for freq in spec:
            spec_drop.append([freq[i] for i in range(len(freq)) if i % 10 == 0])
        return spec_drop

    def make_one_hot(self, samples):
        labels = [samples[i][0] for i in range(len(samples))]
        indexes = [self.classes.index(f) for f in labels]
        one_hot = np.zeros([len(samples),9])
        for i, index_value in enumerate(indexes):
            one_hot[i][index_value] = 1
        return one_hot

    def remove_label(self, samples):
        return [samples[i][1] for i in range(len(samples))]

# util = Util("../TrainingData/UrbanSound8K_modified_v2/audio/")
# trainX,trainY,valX,valY,testX,testY = util.generate_batch(100)
# print(trainX[0])
# print(type(trainX))
# print(type(trainX[0]))
# print(len(trainX[0][0]),len(valX))
