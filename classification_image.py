import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from data_utils import *

height = 100
width = 100
nClass = 5
nSteps=1500

nFeatures1=16
nFeatures2=32
nNeuronsfc=1024

# Read in data
label, image = getImage("data/train-00000-of-00001", height, width, nClass, grayscale=False)
vlabel, vimage = getImage("data/validation-00000-of-00001", height, width, nClass, grayscale=False)

# Shuffle the data for batch processing
imageBatch, labelBatch = tf.train.shuffle_batch(
    [image, label], 
    batch_size=100,
    capacity=2000,
    min_after_dequeue=1000)

vimageBatch, vlabelBatch = tf.train.shuffle_batch(
    [vimage, vlabel], 
    batch_size=100,
    capacity=2000,
    min_after_dequeue=1000)

# Create placeholders
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, width*height*3])
y_ = tf.placeholder(tf.float32, [None, nClass])
x_image = tf.reshape(x, [-1,width,height,3])

# Create CNN layers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
W_conv1 = weight_variable([5, 5, 3, nFeatures1])
b_conv1 = bias_variable([nFeatures1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
W_conv2 = weight_variable([5, 5, nFeatures1, nFeatures2])
b_conv2 = bias_variable([nFeatures2])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
W_fc1 = weight_variable([int((width/4) * (height/4) * nFeatures2), nNeuronsfc])
b_fc1 = bias_variable([nNeuronsfc])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
h_pool2_flat = tf.reshape(h_pool2, [-1, int((width/4) * (height/4) * nFeatures2)])
h_fc1 = tf.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
W_fc2 = weight_variable([nNeuronsfc, nClass])
b_fc2 = bias_variable([nClass])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Compute cost
loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
optimizer = tf.train.AdamOptimizer(1e-4)
train_op = optimizer.minimize(loss)

# Initialize TF session
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

# Start training
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

for i in range(nSteps):
    batch_xs, batch_ys = sess.run([imageBatch, labelBatch])
    cur_prediction = prediction.eval(feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0})
    print(batch_ys.shape)
    train_op.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

    # Do validation every 5 steps
    if (i+1)%5 == 0:
        vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
        train_accuracy = accuracy.eval(feed_dict={x:vbatch_xs, y_: vbatch_ys, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i+1, train_accuracy))

# Stop training 
coord.request_stop()
coord.join(threads)
