import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from data_generator import Data

tf.set_random_seed(1)
np.random.seed(1)
nSteps=200

# Introduce data
X = None
Y = None

# Feed in data
data = Data()
data.random_normal(20, 2, one_hot=True)
X = data.X
Y = data.Y
print(X)
print(Y)

# Plot data
plt.scatter(X[:, 0], X[:, 1], c=[x.argmax(0) for x in Y], s=20, cmap='RdYlGn')
plt.show()

# Create placeholders
tf_x = tf.placeholder(tf.float32, X.shape)
tf_y = tf.placeholder(tf.int32, Y.shape)

# Create neural network layers
l1 = tf.layers.dense(tf_x, 5, tf.nn.relu)
output = tf.layers.dense(l1, 2)

# Compute cost
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
prediction = tf.equal(tf.argmax(tf_y,1), tf.argmax(output,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_op = optimizer.minimize(loss)

# Initialize TF session
sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

# Start training
plt.ion()
plt.show()

for step in range(nSteps):
    # train and net output
    _, acc, pred = sess.run([train_op, accuracy, output], feed_dict={tf_x: X, tf_y: Y})

    if step % 2 == 0:
        plt.cla()
        plt.scatter(X[:, 0], X[:, 1], c=pred.argmax(1), s=20, cmap='RdYlGn')
        plt.text(1.5, -4, 'Accuracy=%.2f' % acc, fontdict={'size': 12, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
