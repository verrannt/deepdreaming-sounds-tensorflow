import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.mlab
from os import listdir
from os.path import isfile, join, isdir
from scipy.io import wavfile

class Util():

    def generate_batch(self, batch_size, path, sample_length):

        ### TODO AT THE MOMENT WE ARE ALWAYS USING THE COMPLETE SAMPLE LENGTH

        #batch_size=100
        #path="../TrainingData/UrbanSound8K_modified_v2/audio/"

        samples_per_class = batch_size // 9
        assert samples_per_class <= 120, "Too many samples per class, max 120. \
            Reduce batch size!"

        trainX, valX, testX = [[] for _ in range(3)]
        train_size, val_size, test_size = [int(samples_per_class*i) for i in [0.6, 0.2, 0.2]]
        self.classes = [f for f in listdir(path) if isdir(join(path,f))]

        for self.class_name in self.classes:
            for _ in range(train_size):
                X = get_random_spec(self,"TRAIN")
                trainX.append([self.class_name, X])
            for _ in range(val_size):
                X = get_random_spec(self,"VALIDATE")
                valX.append([self.class_name, X])
            for _ in range(test_size):
                X = get_random_spec(self,"TEST")
                trainX.append([self.class_name, X])

        np.random.shuffle(trainX)
        np.random.shuffle(valX)
        np.random.shuffle(testX)

        trainY = make_one_hot(trainX)
        valY = make_one_hot(valX)
        testY = make_one_hot(testY)

        return trainX, trainY, valX, valY, testX, testY

    def get_random_spec(self, set):
        all_files = listdir(join(path,self.class_name))
        suitable_files = [f for f in all_files if f.split("-")[0] == set]
        random_file = random.choice(suitable_files)
        wav_file = wavfile.read(join(path+self.class_name, random_file))
        wav_file_n = wav_file/np.max(np.abs(wav_file),axis=0)
        spec = matplotlib.mlab.specgram(wav_file_n)[0]
        return spec

    def make_one_hot(self, samples):
        labels = samples[:,0]
        indexes = [self.classes.index(f) for f in labels]
        one_hot = np.zeros([len(samples),9])
        for i, index_value in enumerate(indexes):
            one_hot[i][index_value] = 1
        return one_hot

U = Util()
trainX, trainY, valX, valY, testX, testY = U.generate_batch(batch_size=100,path="../TrainingData/UrbanSound8K_modified_v2/audio/",sample_length=1)
