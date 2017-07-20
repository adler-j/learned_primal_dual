"""Reference FBP reconstruction for human data."""

import numpy as np
np.random.seed(0)
import odl
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

mu_water = 0.02
photons_per_pixel = 10000.0

epsilon = 1.0 / photons_per_pixel

# Create ODL data structures
size = 512
space = odl.uniform_discr([-128, -128], [128, 128], [size, size],
                          dtype='float32', weighting='const')

# Tomography
# Make a fan beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
angle_partition = odl.uniform_partition(0, 2 * np.pi, 1000)
# Detector: uniformly sampled, n = 558, min = -30, max = 30
detector_partition = odl.uniform_partition(-360, 360, 1000)
geometry = odl.tomo.FanFlatGeometry(angle_partition, detector_partition,
                                    src_radius=500, det_radius=500)


operator = odl.tomo.RayTransform(space, geometry)
pseudoinverse = odl.tomo.fbp_op(operator)

# Create nonlinear forward operator using composition
nonlinear_operator = odl.ufunc_ops.exp(operator.range) * (- mu_water * operator)

# --- Generate artificial data --- #

folder = ...   # Folder to data
phantom = space.element(np.rot90(np.load(folder + 'L286_FD_3_1.CT.0002.0201.2015.12.22.18.22.49.651226.358225786.npy'), -1))

phantom /= 1000.0  # convert go g/cm^3

data = nonlinear_operator(phantom)
data = odl.phantom.poisson_noise(data * photons_per_pixel) / photons_per_pixel

initial = pseudoinverse(-np.log(epsilon + data) / mu_water)

smoothed_pseudoinverse = odl.tomo.fbp_op(operator,
                                         filter_type='Hann',
                                         frequency_scaling=0.45)
recon = smoothed_pseudoinverse(-np.log(epsilon + data) / mu_water)

print('ssim = {}'.format(ssim(phantom.asarray(), recon.asarray())))
print('psnr = {}'.format(psnr(phantom.asarray(), recon.asarray(), dynamic_range=np.max(phantom) - np.min(phantom))))

figure_folder = ''
phantom.show('', clim=[0.8, 1.2], saveto=figure_folder + 'head_phantom')
phantom.show('', coords=[[-40, 25], [-25, 25]], clim=[0.8, 1.2], saveto=figure_folder + 'head_phantom_middle')
data.show('', saveto=figure_folder + 'head_data')
initial.show('', clim=[0.8, 1.2], saveto=figure_folder + 'head_initial')
recon.show('', clim=[0.8, 1.2], saveto=figure_folder + 'head_fbp')
recon.show('', coords=[[-40, 25], [-25, 25]], clim=[0.8, 1.2], saveto=figure_folder + 'head_fbp_middle')