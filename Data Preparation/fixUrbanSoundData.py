'''
We needed to fix the structure of the directory containing the Urban Sound Data,
since the folder structure was not according to the labeling
'''

import pandas as pd
import os

csv = pd.read_csv('../../TrainingData/UrbanSound8K/metadata/UrbanSound8K.csv')
classes_list = csv['class'].drop_duplicates().tolist()
sorted_file_list = []

for class_name in classes_list:
    temp = csv[csv['class'] == class_name]
    file_names = temp['slice_file_name'].tolist()
    sorted_file_list.append(file_names)

for i in range(len(sorted_file_list)):
    for file_name in sorted_file_list[i]:
        try:
            os.rename(file_name, classes_list[i]+'/'+file_name)
        except:
            pass
