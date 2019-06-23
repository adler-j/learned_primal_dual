"""Partially learned gradient descent scheme for ellipses."""

import os
import adler
adler.util.gpu.setup_one_gpu()

from adler.tensorflow import prelu, psnr

import tensorflow as tf
import numpy as np
import odl
import odl.contrib.tensorflow
from mayo_util import DATA_FOLDER, FileLoader

np.random.seed(0)
name = 'mayo_learned_primal_dual_log'

sess = tf.InteractiveSession()


# Create ODL data structures
size = 512
space = odl.uniform_discr([-128, -128], [128, 128], [size, size],
                          dtype='float32', weighting=1.0)

# Tomography
# Make a fan beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
angle_partition = odl.uniform_partition(0, 2 * np.pi, 1000)
# Detector: uniformly sampled, n = 558, min = -30, max = 30
detector_partition = odl.uniform_partition(-360, 360, 1000)
geometry = odl.tomo.FanFlatGeometry(angle_partition, detector_partition,
                                    src_radius=500, det_radius=500)


ray_transform = odl.tomo.RayTransform(space, geometry)

opnorm = odl.power_method_opnorm(ray_transform)
operator = (1 / opnorm) * ray_transform

# Create tensorflow layer from odl operator
odl_op_layer = odl.contrib.tensorflow.as_tensorflow_layer(operator,
                                                          'RayTransform')
odl_op_layer_adjoint = odl.contrib.tensorflow.as_tensorflow_layer(operator.adjoint,
                                                                  'RayTransformAdjoint')

# User selected paramters
n_data = 1
n_iter = 10
n_primal = 5
n_dual = 5
mu_water = 0.02
photons_per_pixel = 10000


file_loader = FileLoader(DATA_FOLDER, exclude='L286')


def generate_data(validation=False):
    """Generate a set of random data."""
    n_iter = 1 if validation else n_data

    y_arr = np.empty((n_iter, operator.range.shape[0], operator.range.shape[1], 1), dtype='float32')
    x_true_arr = np.empty((n_iter, space.shape[0], space.shape[1], 1), dtype='float32')

    for i in range(n_iter):
        if validation:
            fi = DATA_FOLDER + 'L286_FD_3_1.CT.0002.0201.2015.12.22.18.22.49.651226.358225786.npy'
        else:
            fi = file_loader.next_file()

        data = np.load(fi)

        phantom = space.element(np.rot90(data, -1))
        phantom /= 1000.0  # convert go g/cm^3

        data = ray_transform(phantom)
        data = np.exp(-data * mu_water)

        noisy_data = odl.phantom.poisson_noise(data * photons_per_pixel)
        noisy_data = np.maximum(noisy_data, 1) / photons_per_pixel

        log_noisy_data = np.log(noisy_data) * (-1 / mu_water) * (1 / opnorm)

        x_true_arr[i, ..., 0] = phantom
        y_arr[i, ..., 0] = log_noisy_data

    return y_arr, x_true_arr


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
        primal = tf.concat([tf.zeros_like(x_true)] * n_primal, axis=-1)
        dual = tf.concat([tf.zeros_like(y_rt)] * n_dual, axis=-1)

    for i in range(n_iter):
        with tf.variable_scope('dual_iterate_{}'.format(i)):
            evalpt = primal[..., 1:2]
            evalop = odl_op_layer(evalpt)
            update = tf.concat([dual, evalop, y_rt], axis=-1)

            update = prelu(apply_conv(update), name='prelu_1')
            update = prelu(apply_conv(update), name='prelu_2')
            update = apply_conv(update, filters=n_dual)
            dual = dual + update

        with tf.variable_scope('primal_iterate_{}'.format(i)):
            evalpt = dual[..., 0:1]
            evalop = odl_op_layer_adjoint(evalpt)
            update = tf.concat([primal, evalop], axis=-1)

            update = prelu(apply_conv(update), name='prelu_1')
            update = prelu(apply_conv(update), name='prelu_2')
            update = apply_conv(update, filters=n_primal)
            primal = primal + update

        primal_values.append(primal)
        dual_values.append(dual)

    x_result = primal[..., 0:1]


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
print(psnr(primal_values_result[-1][0, ..., 0], x_true_arr_validate[0, ..., 0], dynamic_range=np.max(x_true_arr_validate) - np.min(x_true_arr_validate)))


def normalized(val, sign=False):
    if sign:
        val = val * np.sign(np.mean(val))
    return (val - np.mean(val)) / np.std(val)

path = name
for i in range(n_iter):
    vals = primal_values_result[i]
    space.element(vals[..., 0]).show(saveto='{}/x_{}'.format(path, i))
    space.element(vals[..., 0]).show(clim=[0.8, 1.2], saveto='{}/x_windowed_{}'.format(path, i))
    space.element(normalized(primal_values_result[i][..., 1], True)).show(clim=[-3, 3], saveto='{}/x_eval_{}'.format(path, i))
    operator.range.element(normalized(dual_values_result[i][..., 0])).show(clim=[-3, 3], saveto='{}/y_{}'.format(path, i))

    plt.close('all')

el = space.element(primal_values_result[-1][..., 0])
el.show('', coords=[[-40, 25], [-25, 25]], clim=[0.8, 1.2], saveto='{}/x_midle'.format(path))
