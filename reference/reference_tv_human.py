"""Reference TV reconstruction for human data."""

import adler
adler.util.gpu.setup_one_gpu()

import numpy as np
np.random.seed(0)
import odl
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

mu_water = 0.02
photons_per_pixel = 10000.0

epsilon = 0.0001

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
pseudoinverse = odl.tomo.fbp_op(operator,
                                frequency_scaling=0.45,
                                filter_type='Hann')

# Create nonlinear forward operator using composition
nonlinear_operator = odl.ufunc_ops.exp(operator.range) * (- mu_water * operator)

# --- Generate artificial data --- #
folder = ...   # Folder to data
phantom = space.element(np.rot90(np.load(folder + 'L286_FD_3_1.CT.0002.0201.2015.12.22.18.22.49.651226.358225786.npy'), -1))

phantom /= 1000.0  # convert go g/cm^3

data = nonlinear_operator(phantom)
noisy_data = odl.phantom.poisson_noise(data * photons_per_pixel) / photons_per_pixel


# --- Set up the inverse problem --- #


# Initialize gradient operator
gradient = odl.Gradient(space)

# Column vector of two operators
# scaling the operator acts as a pre-conditioner, improving convergence.
op = odl.BroadcastOperator(nonlinear_operator, gradient)

# Do not use the g functional, set it to zero.
g = odl.solvers.ZeroFunctional(op.domain)

# Create functionals for the dual variable

# l2-squared data matching
data_discr = odl.solvers.KullbackLeibler(operator.range, noisy_data)

# Isotropic TV-regularization i.e. the l1-norm
l1_norm = 0.00011 * odl.solvers.GroupL1Norm(gradient.range)

# Combine functionals, order must correspond to the operator K
f = odl.solvers.SeparableSum(data_discr, l1_norm)


# --- Select solver parameters and solve using Chambolle-Pock --- #

# Choose a starting point
x = pseudoinverse(-np.log(epsilon + noisy_data) / mu_water)

# Estimated operator norm to ensure ||K||_2^2 * sigma * tau < 1
op_norm = odl.power_method_opnorm(op.derivative(x))

niter = 1000  # Number of iterations
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 2.0 / op_norm  # Step size for the dual variable
gamma = 0.00

# Pass callback to the solver to display intermediate results
callback = odl.solvers.CallbackPrintIteration()


with odl.util.Timer('runtime of iterative algorithm'):
    # Run the algorithm
    odl.solvers.pdhg(
        x, f, g, op, tau=tau, sigma=sigma, niter=niter, gamma=gamma,
        callback=callback)


print('ssim = {}'.format(ssim(phantom.asarray(), x.asarray())))
print('psnr = {}'.format(psnr(phantom.asarray(), x.asarray(), dynamic_range=np.max(phantom) - np.min(phantom))))

# Display images
figure_folder = ''
x.show('', clim=[0.8, 1.2], saveto=figure_folder + 'head_tv')
x.show('', coords=[[-40, 25], [-25, 25]], clim=[0.8, 1.2], saveto=figure_folder + 'head_tv_midle')