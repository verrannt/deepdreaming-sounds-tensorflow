# CNN Sound Visualization

Attempt to visualize layers/learned representations of CNN trained on sound data using DeepDream approaches. Current __work in progress__.

Github repositories used so far:
+ [sound-cnn](https://github.com/awjuliani/sound-cnn): Simple CNN used for classifying sound data
+ [tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg16.py): Tensorflow implementation of VGG16 architecture

Dataset we are probably going to train on:
+ [Urban Sound 8K Dataset](https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound8k.html)

References we might use later:
+ [Pre-trained models converted for tensorflow](https://github.com/sfujiwara/tfmodel)
+ [CNN visualization tool in TensorFlow](https://github.com/InFoCusp/tf_cnnvis)
+ [DeepDreaming with TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb)
+ [DeepDream Tutorial](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/14_DeepDream.ipynb)

### Changelog:
##### v0.0.1:
  + Implemented simplified VGG16 architecture: CNN with 5 convolutional layers, 2 fully-connected layers. Size of layers specified in *cnn_architectures.py*

##### v0.0.2:
  + Added train and utilities files. Are not suited towards the network.

##### v0.0.3
  + Added code for solving problems with the UrbanSounds8K dataset in the "Data Preparation" directory. The problems included:
    + the dataset was not structured according to the classes of its samples (the samples stem from 10 different classes and were spread seemingly random over 10 folders, not corresponding to the classes).
    + the samples are of differing lengths ranging from 0.05s to 4.03s.
  + For fixing the structure problem we created the _convAndOrder.sh_ and _fixUrbanSoundData.py_ scripts. The dataset is now structured in a suitable way, containing 10 subfolders each corresponding to one class.
  + The duration problem is not solved yet, leaving us two options:
    1. skip the samples that are below e.g. 1s
    2. feed very short samples into the graph

##### v0.0.3b
  + Plotting the durations of all sound samples, we found that most accumulate in the 3.5 to 4 seconds range:
  ![Alt text](https://raw.githubusercontent.com/verrannt/TF_CNN_SoundVis/master/graphics/durations_histogram.png)
  + Upon further investigation, it turned out that the vast majority of samples is of four seconds length (7320 samples to be precise). Hence, we are only going to use these samples.
