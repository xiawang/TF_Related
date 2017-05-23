import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_generator import Data

tf.set_random_seed(1)
np.random.seed(1)

nSteps = 8000
batch = 64
lr_generator = 0.0001
lr_discriminator = 0.0001
num_gen_seeds = 6
num_simu_points = 18
simulate_points = np.vstack([np.linspace(-1, 1, num_simu_points) for _ in range(batch)])

# Plot real distribution range
plt.plot(simulate_points[0], 2 * np.power(simulate_points[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
plt.plot(simulate_points[0], 1 * np.power(simulate_points[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
plt.legend(loc='upper right')
plt.show()

# Initialize data (feed in data during training)
data = Data()

# Create Generator network
with tf.variable_scope('Generator'):
    gen_in = tf.placeholder(tf.float32, [None, num_gen_seeds])
    gen_l1 = tf.layers.dense(gen_in, 128, tf.nn.relu)
    gen_out = tf.layers.dense(gen_l1, num_simu_points)

# Create Discriminator network
with tf.variable_scope('Discriminator'):
    original_dist = tf.placeholder(tf.float32, [None, num_simu_points])
    dis_l0 = tf.layers.dense(original_dist, 128, tf.nn.relu, name='l')
    prob_origin = tf.layers.dense(dis_l0, 1, tf.nn.sigmoid, name='out')
    # reuse layers for generator
    dis_l1 = tf.layers.dense(gen_out, 128, tf.nn.relu, name='l', reuse=True)
    prob_generate = tf.layers.dense(dis_l1, 1, tf.nn.sigmoid, name='out', reuse=True)

# Compute cost and optimization
dis_cost = -tf.reduce_mean(tf.log(prob_origin) + tf.log(1-prob_generate))
gen_cost = tf.reduce_mean(tf.log(1-prob_generate))

dis_train_op = tf.train.AdamOptimizer(lr_discriminator).\
    minimize(dis_cost, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
gen_train_op = tf.train.AdamOptimizer(lr_generator).\
    minimize(gen_cost, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

# Initialize TF session
sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

# Start training
plt.ion()
plt.show()

for step in range(nSteps):
    # Feed in data
    data.parabola_distribution(batch, simulate_points)
    dist_batches = data.dist_batches
    gen_seeds = np.random.randn(batch, num_gen_seeds)
    gen_dist, prob_o, dis_score, _, _ = \
        sess.run([gen_out, prob_origin, dis_cost, dis_train_op, gen_train_op], \
            feed_dict={gen_in: gen_seeds, original_dist: dist_batches})

    if step % 50 == 0:
        plt.cla()
        plt.plot(simulate_points[0], gen_dist[0], c='#4AD631', lw=3, label='Generated painting',)
        plt.plot(simulate_points[0], 2 * np.power(simulate_points[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(simulate_points[0], 1 * np.power(simulate_points[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_o.mean(), fontdict={'size': 8})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -dis_score, fontdict={'size': 8})
        plt.ylim((0, 3))
        plt.legend(loc='upper right', fontsize=8)
        plt.draw()
        plt.pause(0.01)

plt.ioff()
plt.show()
