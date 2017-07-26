# Deep Dreaming Sounds In Tensorflow

Up until now there has been a lot of work using Google's Deepdream algorithm to visualize the internal representations of artificial neural networks. But to the best of our knowledge, there has not been any work with neural networks trained on sound data instead of image data. In this project we trained a convolutional neural network on sound data in order to apply the Deepdream algorithm to it.

### Progress

This project is currently a __work in progress__. So far, training accuracies reach about 70 %. The Deepdream algorithm is yet to be applied.

### Dataset

We used the [UrbanSound8K dataset](https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound8k.html) compiled by NYU researchers Justin Salamon, Christopher Jacoby and Juan Pablo Bello that we slightly modified ourselves. 
In order to use the network, please download the modified version manually from Google Drive storage (link provided in the reference section) and save it in your local copy of this repository. After extraction (it's a .zip), you should see the folder *"UrbanSound8K_modified"* in the main directory, which contains the *README.txt* and *FREESOUNDCREDITS.txt* files as well as the *"audio"* directory, in which all the .wav-files reside in their respective class. All of our modifications are explained in the *README.txt*.

### File descriptions

To get an overview over the files in this repository, here is a short description:

+ __datatools__ includes several python and shell scripts we wrote for the modification of the dataset. Since the compiling of the dataset is finished these are not necessary for the training/Deepdream procedure and have been included for completeness only.
+ __images__ includes images used in this readme and some infographics about our current progress.
+ __train.py__ is the main file for training the network. It imports __utilities.py__ which is responsible for creating appropriate training, testing and validation batches from the dataset as well as the Tensorflow implementation of a CNN that we wrote in __model.py__. The batches then will be fed into the model in order to train it on the data.
+ __cnn_architectures.py__ contains a simple helper class that returns python dicitonaries describing the shapes of the different neural network layers to ease playing around with different architectures.
+ __eval.py__ is the script for evaluating the performance of the network after training. It needs a valid protobuf file in order to obtain the network's graph def, which can be generated using the __freeze_graph.py__ script.

More on that under [Usage](#usage)!

### Network Architecture

We used a convolutional neural network with four convolutional, two pooling, two dropout and two feed-forward/fully-connected layers. The graph looks as follows:

![Network graph](https://raw.githubusercontent.com/verrannt/deepdreaming-sounds-tensorflow/master/images/graph_2017-07-26_r2.png)

### Usage

The following additional python libraries are needed:
+ tensorflow
+ scipy
+ numpy
+ matplotlib
+ pickle
+ urllib
+ random

#### Training

In order to train the network, navigate to the directory you cloned the repository in. From there, you need to run _train.py_ using an installation of Python 3. In order to work, _train.py_ needs the dataset described above which you should download and save as described. Furthermore, you can specify the number of iterations, batch size and path to the dataset with sys arguments.

```bash
python3 train.py 'number_of_iterations' 'batch_size' 'path'
```

Appropriate usages are to either give only the first sys argument, the first two, all three or none at all. The defaults for arguments not given are:

```python
number_of_iterations = 5000
batch_size = 100
path = "./UrbanSound8K_modified/urbansound.pkl"
```

The meta graph, called *"graph.pb"*, will be saved in *"./logs/model/"*, as will be the *model.ckpt* checkpoint files for the weights. Summary files that can be used for visualizations using tensorboard will be saved in *"./logs/tensorboard/"*. Don't worry if you do not see the logs directory yet, it will be created upon the first call of *train.py*, since git does not allow the syncing of empty directories.

#### Evaluation

We provided a script called *eval.py* which, after successfull training, can be used to further test the trained network on some samples. It needs a valid protobuf file that contains both the network structure (the meta graph) and the saved weights to work. To obtain such a file, use the [*freeze_graph.py*](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py) script made by the Google developers. From this directory, an appropriate call from the shell looks like this:

```bash
python freeze_graph.py --input_graph logs/model/graph.pb --input_checkpoint logs/model/model.ckpt --input_binary True --output_graph logs/model/output.pb --output_node_names fc2/fully_connected/fc_out
```

The *freeze_graph.py* script takes the *graph.pb* meta graph protobuf file and the *model.ckpt* checkpoint files that were generated during the training process as input and outputs the *output.pb* protobuf file that contains both the meta graph and the variable values from the checkpoints in one file. In addition to that, the output_node_name needs to be specified; our graph's output is simply the second feed-forward layer without an activation function, which is called "fc2/fully_connected/fc_out" due to the nature of Tensorflow's variable scopes .

*eval.py* imports the model from the generated *output.pb* file. Its usage is pretty simple, just make sure you have the appropriate file present in the *"./logs/model/"* directory. On your local machine's console, type:

```bash
python3 eval.py 
```

and it will print the results.

#### Dreaming Deep

As mentioned above, the deep dreaming process is not yet applied. 
---

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
  ![Alt text](https://raw.githubusercontent.com/verrannt/TF_CNN_SoundVis/master/images/durations_histogram.png)
  + Upon further investigation, it turned out that the vast majority of samples is of four seconds length (7320 samples to be precise). Hence, we are only going to use these samples.

##### v0.0.3c
  + Added the _deleteDurations.py_ script which removes all samples that are not exactly four seconds long. This leaves us with ~84% of the training data. Yet we found that the "gun_shot" class consists of mainly sub-4-seconds samples (>95%), so we decided to drop that class out of the classification task.
