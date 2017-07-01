import os
from os.path import isfile
import random
import sys

if len(sys.argv) != 5:
	print("Usage: 'python shuffle_files.py mode path training_samples validation_samples'")
	exit(1)

mode = sys.argv[1]
p = sys.argv[2]
t = sys.argv[3]
v = sys.argv[4]

#shuffle assigns TRAIN VALIDATE TEST flags to a specified
#number of files randomly
#train, validate specify how many files should be in each dataset
#the rest will be assigned to testing
def shuffle(path, train, validate):

	files = os.listdir(path)
	random.shuffle(files)
	print(files)

	if (train+validate+validate*0.5 > len(files)):
		print("There are not enough files for the parameters!")
		exit(1)

	training_set = files[:train]
	validation_set = files[train:train + validate]
	testing_set = files[validate+train:]

	for file in training_set:
		if isfile(path+file):
			os.rename(os.path.join(path,file),os.path.join(path,"TRAIN-"+file))
	for file in validation_set:
		if isfile(path+file):
			os.rename(os.path.join(path,file),os.path.join(path,"VALIDATE-"+file))
	for file in testing_set:
		if isfile(path+file):
			os.rename(os.path.join(path,file),os.path.join(path,"TEST-"+file))


#removes flags set by shuffle()
def unshuffle(path):

	files = os.listdir(path)

	for file in files:

		if any(a ==  str(file).split("-")[0] for a in ["TRAIN", "TEST", "VALIDATE"] ):
			os.rename(os.path.join(path,file) , os.path.join(path , str(file).split("-",1)[1]))


if mode == "shuffle":
	shuffle(p,int(t),int(v))
elif mode == "unshuffle":
	unshuffle(p)
else:
	print("Please use as 'python shuffle_files.py mode path training_samples validation_samples'")
