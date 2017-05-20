import wave
import contextlib
import sys
import os
from os import listdir
from os.path import isfile, isdir, join
import matplotlib.pyplot as plt

# if sys input is given to specify directory, take it
try:
    mypath = sys.argv[1]
# else use this
except IndexError:
    mypath = "../../TrainingData/UrbanSound8K/audio/"

# Get list of all directories/classes
dirs = [f for f in listdir(mypath) if isdir(join(mypath, f))]

durations = []

for dir_name in dirs:
    # adjust path variable
    path = mypath + dir_name + "/processed/"
    # Get list of all files in directory
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    for fname in onlyfiles:
        if fname.split(".")[-1] == "wav":
            f = wave.open(path+fname,"r")
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            durations.append(duration)

print("Longest duration: " + str(max(durations)))
print("Shortest duration: " + str(min(durations)))
print("How often 4 seconds: " + str(durations.count(4.0)))

plt.hist(durations,bins=8)
plt.show()
