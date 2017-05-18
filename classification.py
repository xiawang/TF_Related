import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

# Feed in data
X = None
Y = None

# experiment data
n_data = np.ones((20, 2))
x0 = np.random.normal(2*n_data, 1)
y0 = np.zeros(20)
x1 = np.random.normal(-2*n_data, 1)
y1 = np.ones(20)
X = np.vstack((x0, x1))
Y = np.hstack((y0, y1))

print(type(X))
print(X)

# plot data
plt.scatter(X[:, 0], X[:, 1], c=Y, s=20, cmap='RdYlGn')
plt.show()

tf_x = tf.placeholder(tf.float32, X.shape)
tf_y = tf.placeholder(tf.int32, Y.shape)

# neural network layers
l1 = tf.layers.dense(tf_x, 5, tf.nn.relu)
output = tf.layers.dense(l1, 2)

# compute cost
loss = tf.losses.sparse_softmax_cross_entropy(labels=tf_y, logits=output)
accuracy = tf.metrics.accuracy(labels=tf.squeeze(tf_y), predictions=tf.argmax(output, axis=1),)[1]
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_op = optimizer.minimize(loss)

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

plt.ion()
plt.show()
for step in range(100):
    # train and net output
    _, acc, pred = sess.run([train_op, accuracy, output], {tf_x: X, tf_y: Y})

    if step % 2 == 0:
        plt.cla()
        plt.scatter(X[:, 0], X[:, 1], c=pred.argmax(1), s=20, cmap='RdYlGn')
        plt.text(1.5, -4, 'Accuracy=%.2f' % acc, fontdict={'size': 12, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
