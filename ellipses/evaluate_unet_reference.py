import adler
adler.util.gpu.setup_one_gpu()

from adler.odl.phantom import random_phantom
from adler.tensorflow import prelu, cosine_decay, reference_unet

import tensorflow as tf
import numpy as np
import odl
import odl.contrib.tensorflow

np.random.seed(0)
name = 'unet_reference'

sess = tf.InteractiveSession()

# Create ODL data structures
size = 128
space = odl.uniform_discr([-64, -64], [64, 64], [size, size],
                          dtype='float32')


geometry = odl.tomo.parallel_beam_geometry(space, num_angles=30)
ray_trafo = odl.tomo.RayTransform(space, geometry)
pseudoinverse = odl.tomo.fbp_op(ray_trafo,
                                filter_type='Hann')

# Ensure operator has fixed operator norm for scale invariance
opnorm = odl.power_method_opnorm(ray_trafo)
pseudoinverse = pseudoinverse * opnorm
operator = (1 / opnorm) * ray_trafo

# User selected paramters
n_data = 5

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


with tf.name_scope('correction'):
    dx = reference_unet(x_0, 1,
                        ndim=2,
                        features=64,
                        keep_prob=1.0,
                        use_batch_norm=False,
                        activation='relu',
                        is_training=is_training,
                        name='unet_dx')

    x_result = x_0 + dx


# Initialize all TF variables
sess.run(tf.global_variables_initializer())

# Add op to save and restore
saver = tf.train.Saver()

if 0:
    saver.restore(sess,
                  adler.tensorflow.util.default_checkpoint_path(name))

# Generate validation data
x_arr_validate, x_true_arr_validate = generate_data(validation=True)

with odl.util.Timer():
    x_result_result = sess.run(x_result,
                          feed_dict={x_true: x_true_arr_validate,
                                     x_0: x_arr_validate,
                                     is_training: False})

import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

print(ssim(x_result_result[0, ..., 0], x_true_arr_validate[0, ..., 0]))
print(psnr(x_result_result[0, ..., 0], x_true_arr_validate[0, ..., 0], dynamic_range=1))

path = name
space.element(x_result_result[..., 0]).show(clim=[0, 1], saveto='{}/x'.format(path))
space.element(x_result_result[..., 0]).show(clim=[0.1, 0.4], saveto='{}/x_windowed'.format(path))
plt.close('all')
