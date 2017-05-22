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

    def generate_batch(self, batch_size, sample_length):

        ### TODO AT THE MOMENT WE ARE ALWAYS USING THE COMPLETE SAMPLE LENGTH

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

        trainY = self.make_one_hot(trainX)
        valY = self.make_one_hot(valX)
        testY = self.make_one_hot(testX)

        return trainX, trainY, valX, valY, testX, testY

    def get_random_spec(self, set):
        all_files = listdir(join(self.path, self.class_name))
        suitable_files = [f for f in all_files if f.split("-")[0] == set]
        random_file = random.choice(suitable_files)
        wav_file = wavfile.read(join(self.path + self.class_name, random_file))[1]
        wav_file = wav_file[8000:24000]
        print(len(wav_file))
        wav_file = wav_file/np.max(np.abs(wav_file),axis=0)
        spec = matplotlib.mlab.specgram(wav_file)[0]
        return spec

    def make_one_hot(self, samples):
        labels = []
        for i in range(len(samples)):
            labels.append(samples[i][0])
        indexes = [self.classes.index(f) for f in labels]
        one_hot = np.zeros([len(samples),9])
        for i, index_value in enumerate(indexes):
            one_hot[i][index_value] = 1
        return one_hot

U = Util("../TrainingData/UrbanSound8K_modified_v2/audio/")
trainX, trainY, valX, valY, testX, testY = U.generate_batch(batch_size=150, sample_length=1)
# print(testX,testY)
print(len(trainX[0][1]),len(trainY))
print(len(valX), len(valY))
print(len(testX),len(testY))
