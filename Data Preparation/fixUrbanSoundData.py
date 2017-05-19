'''
We needed to fix the structure of the directory containing the Urban Sound Data,
since the folder structure was not according to the class labeling

Usage: run this script from the directory where all Urban Sound Datasamples
are (without the ten given subfolders),

@date 2017-05-19
'''

import pandas as pd
import os
import sys

path_to_csv = sys.argv[1]
#'../../TrainingData/UrbanSound8K/metadata/UrbanSound8K.csv'
csv = pd.read_csv(path_to_csv)
# contains all class titles as strings
classes_list = csv['class'].drop_duplicates().tolist()
# will contain all file names according to one class
sorted_file_list = []

for class_name in classes_list:
    temp = csv[csv['class'] == class_name]
    # the file names of one class
    file_names = temp['slice_file_name'].tolist()
    sorted_file_list.append(file_names)

for i in range(len(sorted_file_list)):
    for file_name in sorted_file_list[i]:
        try:
            os.rename(file_name, classes_list[i]+'/'+file_name)
        except:
            pass
