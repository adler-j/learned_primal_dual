"""Partially learned gradient descent scheme for ellipses."""

import os
import adler
adler.util.gpu.setup_one_gpu()

from adler.odl.phantom import random_phantom
from adler.tensorflow import prelu, cosine_decay

import tensorflow as tf
import numpy as np
import odl

np.random.seed(0)
name = 'learned_primal_noop'

sess = tf.InteractiveSession()

# Create ODL data structures
size = 128
space = odl.uniform_discr([-64, -64], [64, 64], [size, size],
                          dtype='float32')


geometry = odl.tomo.parallel_beam_geometry(space, angles=30)
ray_trafo = odl.tomo.RayTransform(space, geometry)
pseudoinverse = odl.tomo.fbp_op(ray_trafo)

# Ensure operator has fixed operator norm for scale invariance
opnorm = odl.power_method_opnorm(ray_trafo)
pseudoinverse = pseudoinverse * opnorm
operator = (1 / opnorm) * ray_trafo

# User selected paramters
n_data = 5
n_primal = 5
n_iter = 10

def generate_data(validation=False):
    """Generate a set of random data."""
    n_generate = 1 if validation else n_data

    x_arr = np.empty((n_generate, space.shape[0], space.shape[1], 1), dtype='float32')
    x_true_arr = np.empty((n_generate, space.shape[0], space.shape[1], 1), dtype='float32')

    for i in range(n_generate):
        if validation:
            phantom = odl.phantom.shepp_logan(space, True)
        else:
            phantom = random_phantom(space)
        data = operator(phantom)
        noisy_data = data + odl.phantom.white_noise(operator.range) * np.mean(np.abs(data)) * 0.05
        fbp = pseudoinverse(noisy_data)

        x_arr[i, ..., 0] = fbp
        x_true_arr[i, ..., 0] = phantom

    return x_arr, x_true_arr


with tf.name_scope('placeholders'):
    x_0 = tf.placeholder(tf.float32, shape=[None, size, size, 1], name="x_0")
    x_true = tf.placeholder(tf.float32, shape=[None, size, size, 1], name="x_true")
    is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

def apply_conv(x, filters=32):
    return tf.layers.conv2d(x, filters=filters, kernel_size=3, padding='SAME',
                            kernel_initializer=tf.contrib.layers.xavier_initializer())

primal_values = []

with tf.name_scope('tomography'):
    with tf.name_scope('initial_values'):
        primal = tf.concat([x_0] * n_primal, axis=-1)
    for i in range(n_iter):
        with tf.variable_scope('primal_iterate_{}'.format(i)):
            update = prelu(apply_conv(primal), name='prelu_1')
            update = prelu(apply_conv(update), name='prelu_2')
            update = apply_conv(update, filters=n_primal)
            primal = primal + update

        primal_values.append(primal)

    x_result = primal[..., 0:1]


# Initialize all TF variables
sess.run(tf.global_variables_initializer())

# Add op to save and restore
saver = tf.train.Saver()

if 1:
    saver.restore(sess,
                  adler.tensorflow.util.default_checkpoint_path(name))

# Generate validation data
x_arr_validate, x_true_arr_validate = generate_data(validation=True)


primal_values_result = sess.run(primal_values,
                      feed_dict={x_true: x_true_arr_validate,
                                 x_0: x_arr_validate,
                                 is_training: False})

import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

print(ssim(primal_values_result[-1][0, ..., 0], x_true_arr_validate[0, ..., 0]))
print(psnr(primal_values_result[-1][0, ..., 0], x_true_arr_validate[0, ..., 0], dynamic_range=1))
raise Exception

path = name
for i in range(n_iter):
    space.element(primal_values_result[i][..., 0]).show(clim=[0, 1], saveto='{}/x_{}'.format(path, i))
    space.element(primal_values_result[i][..., 0]).show(clim=[0.1, 0.4], saveto='{}/x_windowed_{}'.format(path, i))
    plt.close('all')