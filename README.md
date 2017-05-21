# TensorFlow Experiments

### Classification

### RNN & LSTM

### GAN & Conditional GAN

### Data Handling
[General Introduction for Reading Data in TensorFlow](https://www.tensorflow.org/programmers_guide/reading_data)
[Inputs and Readers in TensorFlow](https://www.tensorflow.org/api_guides/python/io_ops#Readerss)
#### TFRecord (handling images)
Most easy-to-read TensorFlow examples online use MNIST data set, and there is a simple load function for the MNIST. However, it is hard to find an instruction on how to read in own images for classification or other tasks. In general, for TensorFlow, a single image is viewed as a numpy array containing **rgb** values for each pixel. This numpy array is often reduced to be 1-demensional with the size `width * height * 3`, and in the order of `[r,g,b,r,g,b,r,g,b...]`. Each value in the arry is a floating point number between 0 and 1 (thus not in the normal scalue of 0 to 255). Labels used by TensorFlow are often in one-hot encoding (e.g. if we have 5 classes, and a given example is in the first class, its label is `[1,0,0,0,0]`).

It is true that we can write our own data processing function (pipeline) to convert images to arrays in the format mentioned above, since TensorFlow provides us its own handler and data format, it would be even more efficient just to use its native solutions. Thus, we will introduce the way of using **build_image_data.py** to convert image data repositories to the **TFRecords file**.

#### CSV

### Prerequisite

```
python3
tensorflow v1.x
numpy
matplotlib
```