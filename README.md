# CNN Sound Visualization

Attempt to visualize layers/learned representations of CNN trained on sound data using DeepDream approaches.
Current __work in progress__. Public only because I'm not paying for github. <3

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
  + Implemented simplified VGG16 architecture: CNN with 5 convolutional layers, 2 fully-connected layers.

##### v0.0.2:
  + Added train and utilities files. Are not suited towards the network.
  + Added data preparation files for sorting Urban Sound Dataset.
