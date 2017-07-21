'''
Splits dataset into samples of specified duration (we used 4 seconds) and
puts all samples not of this duration into seperate directory

@date 2017-05-19
'''

import os
import wave
import sys
from os import listdir
from os.path import isfile, isdir, join

# Hyperparameter
wantedduration = sys.argv[1]

# if sys input is given to specify directory, take it
try:
    mypath = sys.argv[2]
# else use current
except IndexError:
    mypath = "./"

try:
    os.mkdir(mypath+"no4secs")
except:
    pass

path_processed = mypath+"processed/"
onlyfiles = [f for f in listdir(path_processed) if isfile(join(path_processed, f))]
for file_name in onlyfiles:
    if file_name.split(".")[-1] == "wav":
        f = wave.open(path_processed+file_name,"r")
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        if str(duration) != wantedduration:
            os.rename(path_processed+file_name, mypath+'no4secs/'+file_name)
