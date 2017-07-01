import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.mlab
import pickle
from os import listdir
from os.path import isfile, join, isdir
from scipy.io import wavfile

class Util():

    def __init__(self, path):
        self.path = path # '../TrainingData/UrbanSound8K_modified_v2/audio/'
        # self.classes = [f for f in listdir(self.path) if isdir(join(self.path,f))]
        self.classes = ['air_conditioner', 'car_horn', 'children_playing',
            'dog_bark', 'drilling', 'engine_idling', 'jackhammer', 'siren',
            'street_music']
        self.threshold_factor = 50

    def generate_batch_from_pickle(self, batch_size):
        ''' generates batch from urbansound.pkl pickle file '''

        # SAMPLE LENGTH IS CURRENTLY FIXATED TO 16,000 DATAPOINTS

        with open(self.path + 'urbansound.pkl', 'rb') as f:
            whole_dataset = pickle.load(f)

        train_data, val_data, test_data = whole_dataset
        samples_per_class = batch_size // 9
        assert samples_per_class <= 120, "Too many samples per class, max 120. \
            Reduce batch size!"

        trainX, valX, testX = [], [], []
        train_size = samples_per_class
        val_size  = int(0.3*train_size) # [int(samples_per_class*i) for i in [0.6, 0.2, 0.2]]
        test_size = int(0.3*train_size)

        for class_name in self.classes:
            class_data = [f for f in train_data if f[0] == class_name]
            for _ in range(train_size):
                random_file = random.choice(class_data)
                s = self.spectogram(random_file[1])
                trainX.append([random_file[0], s])
            class_data = [f for f in val_data if f[0] == class_name]
            for _ in range(val_size):
                random_file = random.choice(class_data)
                s = self.spectogram(random_file[1])
                valX.append([random_file[0], s])
            class_data = [f for f in test_data if f[0] == class_name]
            for _ in range(test_size):
                random_file = random.choice(class_data)
                s = self.spectogram(random_file[1])
                testX.append([random_file[0], s])

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

    def generate_batch_from_wav(self, batch_size):
        ''' generates batch from directory structure containing wav files '''

        # SAMPLE LENGTH IS CURRENTLY FIXATED TO 16,000 DATAPOINTS

        samples_per_class = batch_size // 9
        assert samples_per_class <= 120, "Too many samples per class, max 120. \
            Reduce batch size!"

        trainX, valX, testX = [], [], []
        train_size = samples_per_class
        val_size  = int(0.3*train_size) # [int(samples_per_class*i) for i in [0.6, 0.2, 0.2]]
        test_size = int(0.3*train_size)

        for self.class_name in self.classes:
            for _ in range(train_size):
                s = self.get_random("TRAIN")
                trainX.append([self.class_name, s])
            for _ in range(val_size):
                s = self.get_random("VALIDATE")
                valX.append([self.class_name, s])
            for _ in range(test_size):
                s = self.get_random("TEST")
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

    def get_random(self, set):
        all_files = listdir(join(self.path, self.class_name))
        suitable_files = [f for f in all_files if f.split("-")[0] == set]
        random_file = random.choice(suitable_files)
        wav_file = wavfile.read(join(self.path + self.class_name, random_file))[1]
        spec = self.spectogram(wav_file)
        return spec

    def spectogram(self, wav_file):
        wav_file = wav_file[8000:24000]
        spec = matplotlib.mlab.specgram(wav_file)[0]
        spec = self.drop_timesteps(spec)
        spec = self.sparse_sample(spec)
        spec = self.norm(spec)
        #spec = np.transpose(spec)
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

    def norm(self, spec):
        ''' z-transformation of the input spectrogram '''
        spec_norm = (spec - np.mean(spec) / np.std(spec))
        # for freq in spec:
        #     for i in range(len(freq)):
        #         freq[i] - np.mean
        #     spec_norm.append([freq[i] for i in range(len(freq)))
        #     for freq_spectrum in sample:
        #         freq_spectrum = (freq_spectrum-np.mean(sample))/np.std(sample)
        return spec_norm

# util = Util("../TrainingData/UrbanSound8K_modified_v2/pickle/")
# import os
# os.getcwd()
# trainX,trainY,valX,valY,testX,testY = util.generate_batch_from_pickle(100)
# print(trainX[0].shape)
# print(type(trainX))
# print(type(trainX[0]))
# print(len(trainX[0][0]),len(valX))
