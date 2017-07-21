'''
We needed to fix the structure of the directory containing the Urban Sound Data,
since the folder structure was not according to the class labeling.s

**IMPORTANT NOTE**: In order for this script to work, the files need to already
be extracted from the random folder architecture so that this script runs in a
directory that contains ALL the sound samples at once.
The subdirectories labeled according to the classes need to be established as well.

@date 2017-05-19
'''

import pandas as pd
import os
import sys

# Get csv file
try:
    path_to_csv = sys.argv[1]
except IndexError:
    path_to_csv = "./UrbanSound8K.csv"
csv = pd.read_csv(path_to_csv)

# contains all ten class titles as strings
classes_list = csv['class'].drop_duplicates().tolist()
# will contain all file names for each class
sorted_file_list = []

# for each class
for class_name in classes_list:
    # make list of the file names of one class
    temp = csv[csv['class'] == class_name]
    file_names = temp['slice_file_name'].tolist()
    # and add them to the list of sorted file names
    sorted_file_list.append(file_names)

# for each class
for i in range(len(sorted_file_list)):
    # go through file names of this class
    for file_name in sorted_file_list[i]:
        # and put them into subdirectory that correspond to its class
        try:
            os.rename(file_name, classes_list[i]+'/'+file_name)
        except:
            pass
