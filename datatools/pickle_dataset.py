'''
Simple script that reads .wav-files from the UrbanSound8K_modified_v2 dataset
into a python array using scipy's wavfile library and stores them in three
labeled python arrays for TRAINING, VALIDATION, and TESTING. Then it exports
these using pickle.

@date 2017-07-01
'''

import sys
import pickle
from scipy.io import wavfile
from os import listdir
from os.path import join, isdir


path = "./"
classes = [f for f in listdir(path) if isdir(join(path,f))]
classes = sorted(classes)

TRAIN_DATA, VAL_DATA, TEST_DATA = [], [], []

for class_name in classes:
    all_files = listdir(join(path, class_name))
    for file_name in all_files:
        if file_name.split("-")[0] == "TRAIN":
            wav_file = wavfile.read(join(path + class_name, file_name))[1]
            TRAIN_DATA.append([class_name, wav_file])
        if file_name.split("-")[0] == "VALIDATE":
            wav_file = wavfile.read(join(path + class_name, file_name))[1]
            VAL_DATA.append([class_name, wav_file])
        if file_name.split("-")[0] == "TEST":
            wav_file = wavfile.read(join(path + class_name, file_name))[1]
            TEST_DATA.append([class_name, wav_file])

export = [TRAIN_DATA, VAL_DATA, TEST_DATA]

with open('urbansound.pkl', 'wb') as f:
    pickle.dump(export, f)
