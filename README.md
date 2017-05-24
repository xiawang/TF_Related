# TensorFlow Experiments

### Basic Classification & CNN
Currently, two examples are provided for general classification problem. 

[**classification.py**](https://github.com/xiawang/TF_Related/blob/master/classification.py) clearly shows the general working pipeline for TensorFlow. It provides an example showing how to classify labeled points in a 2-D surface (Cartesian coordinate system) using a single layer neural network. The data used in this example is generated via random distribution function in numpy.

[**classification_image.py**](https://github.com/xiawang/TF_Related/blob/master/classification_image.py) shows a more comlicated example for a training pipeline that deals with image classification. The data used in this example is generated via [**build_image_data.py**](https://github.com/tensorflow/models/blob/master/inception/inception/data/build_image_data.py), and the program is directly handling converted **TFRecord** files. (The detail of converting directory of images to **TFRecord** format is illustrated below.)

Note that in both examples, labels are created using **one-hot** encoding.

### RNN & LSTM
(To be completed...)

### GAN & Conditional GAN
[**Original TensorFlow Codes by MorvanZhou**](https://github.com/MorvanZhou)

#### Basic Usage (adapted)
Generative Adversarial Networks (GAN) is used to do unsupervised learning (generate distributions), and is quite different from the neural network used in classification tasks. The Basic structure of a GAN contains a Generator and a Discriminator. Generator starts from some random noises (random values stored in some number of nodes), and goes through some layers of networks, and finally outputs a distribution (some number of values). While Discriminator will read in both a real distribution and a distribution from the Generator, carry both distributions through some layers of networks, and output two values: probabilities for real as well as generated distributions to be real. Of course, in the optimizer, both `-(log(prob_real) + log(1-prob_generated))` (for the discriminator)  and `log(1-prob_generated)` (for the generator) are minimized.

The Conditional-GAN is very similar to the GAN. The only difference is that an extra tensor (that contains conditional information) is concatenated with the original inputs (in both Generator and Discriminator) and then pushed through the network (e.g. using `tf.concat` function).

#### More Advanced Usage
(To be completed...)

### Data Handling
[General Introduction for Reading Data in TensorFlow](https://www.tensorflow.org/programmers_guide/reading_data)

[Inputs and Readers in TensorFlow](https://www.tensorflow.org/api_guides/python/io_ops#Readerss)
#### TFRecord (handling images)
Most easy-to-read TensorFlow examples online use MNIST data set, and there is a simple load function for the MNIST. However, it is hard to find an instruction on how to read in own images for classification or other tasks. In general, for TensorFlow, a single image is viewed as a numpy array containing **rgb** values for each pixel. This numpy array is often reduced to be 1-demensional with the size `width * height * 3`, and in the order of `[r,g,b,r,g,b,r,g,b...]`. Each value in the array is a floating point number between 0 and 1 (thus not in the normal scale of 0 to 255). Labels used by TensorFlow are often in one-hot encoding (e.g. if we have 5 classes, and a given example is in the first class, its label is `[1,0,0,0,0]`).

It is true that we can write our own data processing function (pipeline) to convert images to arrays in the format mentioned above, since TensorFlow provides us its own handler and data format, it would be even more efficient just to use its native solutions. Thus, we will introduce the way of using [**build_image_data.py**](https://github.com/tensorflow/models/blob/master/inception/inception/data/build_image_data.py) to convert image data repositories to the **TFRecords file** (that can be read in directly for training and testing).

**Step1**
Create a directory called `data`, in which create two sub-directories `train` and `validate`.

**Step2**
In both `train` and `validate`, create folders whose names are class names and copy training and validation images in (e.g. in the directory `train`, copy 8 images to folders of all classes, and in `validate`, copy 2 images to folders of all classes).

**Step3**
Put the script **build_image_data.py** into the `data` directory.

**Step4**
Create a **mylabels.txt** file that contains class names (shown below).
```
Apple
Banana
Grape
Orange
Watermelon

```

(A visualization of the directory structure)

![Directory Structure](https://github.com/xiawang/TF_Related/blob/master/img/01.png)

**Step5**
In the `data` directory, execute the following command:
```
python build_image_data.py --train_directory=./train --output_directory=./  \
--validation_directory=./validate --labels_file=mylabels.txt   \
--train_shards=1 --validation_shards=1 --num_threads=1
```

Two TFRecord files for training and validating will be created: `train-00000-of-00001` and `validation-00000-of-00001`. The function for reading in these two files is shown in the file [**data_utils.py**](https://github.com/xiawang/TF_Related/blob/master/data_utils.py). The batching, training and validating process are shown in the file [**classification_image.py**](https://github.com/xiawang/TF_Related/blob/master/classification_image.py).

#### CSV

### Prerequisite

```
python3
tensorflow v1.x
numpy
matplotlib
```

### Useful Resources
* [**Deep NLP (Oxford Online Course)**](https://github.com/oxford-cs-deepnlp-2017/lectures)
* [**RNN in TensorFlow (Tutorials from Erik Hallström)**](https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767)