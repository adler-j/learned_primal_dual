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
name = 'learned_chambolle_pock'

sess = tf.InteractiveSession()

# Create ODL data structures
size = 128
space = odl.uniform_discr([-64, -64], [64, 64], [size, size],
                          dtype='float32')

geometry = odl.tomo.parallel_beam_geometry(space, angles=30)
operator = odl.tomo.RayTransform(space, geometry)

# Ensure operator has fixed operator norm for scale invariance
opnorm = odl.power_method_opnorm(operator)
operator = (1 / opnorm) * operator

# Create tensorflow layer from odl operator
odl_op_layer = odl.contrib.tensorflow.as_tensorflow_layer(operator,
                                                          'RayTransform')
odl_op_layer_adjoint = odl.contrib.tensorflow.as_tensorflow_layer(operator.adjoint,
                                                                  'RayTransformAdjoint')

# User selected paramters
n_data = 5
n_iter = 10

def generate_data(validation=False):
    """Generate a set of random data."""
    n_generate = 1 if validation else n_data

    y_arr = np.empty((n_generate, operator.range.shape[0], operator.range.shape[1], 1), dtype='float32')
    x_true_arr = np.empty((n_generate, space.shape[0], space.shape[1], 1), dtype='float32')

    for i in range(n_generate):
        if validation:
            phantom = odl.phantom.shepp_logan(space, True)
        else:
            phantom = random_phantom(space)
        data = operator(phantom)
        noisy_data = data + odl.phantom.white_noise(operator.range) * np.mean(np.abs(data)) * 0.05

        x_true_arr[i, ..., 0] = phantom
        y_arr[i, ..., 0] = noisy_data

    return y_arr, x_true_arr


with tf.name_scope('variables'):
    sigma = tf.Variable(tf.constant(0.5, dtype=tf.float32), name='sigma')
    tau = tf.Variable(tf.constant(0.5, dtype=tf.float32), name='tau')
    theta = tf.Variable(tf.constant(1, dtype=tf.float32), name='theta')


with tf.name_scope('placeholders'):
    x_true = tf.placeholder(tf.float32, shape=[None, size, size, 1], name="x_true")
    y_rt = tf.placeholder(tf.float32, shape=[None, operator.range.shape[0], operator.range.shape[1], 1], name="y_rt")
    is_training = tf.placeholder(tf.bool, shape=(), name='is_training')


def apply_conv(x, filters=32):
    return tf.layers.conv2d(x, filters=filters, kernel_size=3, padding='SAME',
                            kernel_initializer=tf.contrib.layers.xavier_initializer())


primal_values = []
dual_values = []

with tf.name_scope('tomography'):
    with tf.name_scope('initial_values'):
        primal = tf.zeros_like(x_true)
        primal_bar = tf.zeros_like(x_true)
        dual = tf.zeros_like(y_rt)

    for i in range(n_iter):
        with tf.variable_scope('dual_iterate',
                               reuse=True if i != 0 else None):
            evalop = odl_op_layer(primal_bar)
            update = tf.concat([dual + sigma * evalop, y_rt], axis=-1)

            update = prelu(apply_conv(update), name='prelu_1')
            update = prelu(apply_conv(update), name='prelu_2')
            update = apply_conv(update, filters=1)
            dual = dual + update

        with tf.variable_scope('primal_iterate',
                               reuse=True if i != 0 else None):
            evalop = odl_op_layer_adjoint(dual)
            update = primal - tau * evalop

            update = prelu(apply_conv(update), name='prelu_1')
            update = prelu(apply_conv(update), name='prelu_2')
            update = apply_conv(update, filters=1)
            primal = primal + update

            primal_bar = primal + theta * update

        primal_values.append(tf.concat([primal, primal_bar], axis=-1))
        dual_values.append(dual)

    x_result = primal



# Initialize all TF variables
sess.run(tf.global_variables_initializer())

# Add op to save and restore
saver = tf.train.Saver()

if 1:
    saver.restore(sess,
                  adler.tensorflow.util.default_checkpoint_path(name))

# Generate validation data
y_arr_validate, x_true_arr_validate = generate_data(validation=True)

primal_values_result, dual_values_result = sess.run([primal_values, dual_values],
                      feed_dict={x_true: x_true_arr_validate,
                                 y_rt: y_arr_validate,
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
    space.element(primal_values_result[i][..., 1]).show(saveto='{}/x_eval_{}'.format(path, i))
    operator.range.element(dual_values_result[i][..., 0]).show(saveto='{}/y_{}'.format(path, i))
    plt.close('all')
