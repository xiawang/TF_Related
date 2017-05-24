import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_generator import SequenceData

learning_rate = 0.01
nSteps = 8000
batch_size = 100
display_step = 10
seq_max_len = 20
n_hidden = 50
n_classes = 2

# Generate data
trainset = SequenceData(n_samples=1000, max_seq_len=seq_max_len)
testset = SequenceData(n_samples=500, max_seq_len=seq_max_len)

# Create placeholders
x = tf.placeholder("float", [None, seq_max_len, 1])
y = tf.placeholder("float", [None, n_classes])
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def dynamicRNN(x, seqlen, weights, biases):
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    x = tf.unstack(x, seq_max_len, 1)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    batch_size = tf.shape(outputs)[0]
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    return tf.matmul(outputs, weights['out']) + biases['out']

# Compute prediction
pred = dynamicRNN(x, seqlen, weights, biases)

# Compute cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize TF session
sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

# Start training
for step in range(1, nSteps):
    batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})

    if step % display_step == 0:
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
        print("Step " + str(step) + ", Minibatch Loss= " + \
              "{:.6f}".format(loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc))
print("Optimization Finished!")

# Calculate accuracy
test_data = testset.data
test_label = testset.labels
test_seqlen = testset.seqlen
print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label, seqlen: test_seqlen}))
