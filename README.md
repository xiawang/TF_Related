# TensorFlow Experiments

### Classification

### RNN & LSTM

### GAN & Conditional GAN

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
![Directory Structure](https://github.com/xiawang/TF_Related/tree/master/img)

**Step5**
In the `data` directory, execute the following command:
```
python build_image_data.py --train_directory=./train --output_directory=./  \
--validation_directory=./validate --labels_file=mylabels.txt   \
--train_shards=1 --validation_shards=1 --num_threads=1
```

Two TFRecord files for training and validating will be created: `train-00000-of-00001` and `validation-00000-of-00001`. The functionf for reading in these two files is shown in the file **data_utils.py**.

#### CSV

### Prerequisite

```
python3
tensorflow v1.x
numpy
matplotlib
```