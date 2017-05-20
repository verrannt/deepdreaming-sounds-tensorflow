import wave
import contextlib
import sys
import os
from os import listdir
from os.path import isfile, join

try:
    mypath = sys.argv[1]
except IndexError:
    mypath = "./"

# Get list of all files in directory
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
durations = []

for fname in onlyfiles:
    if fname.split(".")[-1] == "wav":
	#with contextlib.closing(wave.open(mypath+fname,'r')) as f:
        f = wave.open(mypath+fname,"r")
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        durations.append(duration)

maxi = max(durations)
mini = min(durations)
print("Longest duration: " + str(maxi))
print("Shortest duration: " + str(mini))
