"""Reference TV reconstruction for ellipse data."""

import numpy as np
import odl
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

np.random.seed(0)

# Create ODL data structures
size = 128
space = odl.uniform_discr([-64, -64], [64, 64], [size, size],
                          dtype='float32')

geometry = odl.tomo.parallel_beam_geometry(space, num_angles=30)
operator = odl.tomo.RayTransform(space, geometry)
pseudoinverse = odl.tomo.fbp_op(operator)


# --- Generate artificial data --- #


# Create phantom
phantom = odl.phantom.shepp_logan(space, modified=True)

# Create sinogram of forward projected phantom with noise
data = operator(phantom)
data += odl.phantom.white_noise(operator.range) * np.mean(np.abs(data)) * 0.05


# --- Set up the inverse problem --- #


# Initialize gradient operator
gradient = odl.Gradient(space)

# Column vector of two operators
op = odl.BroadcastOperator(operator, gradient)

# Do not use the g functional, set it to zero.
g = odl.solvers.ZeroFunctional(op.domain)

# Create functionals for the dual variable

# l2-squared data matching
l2_norm = odl.solvers.L2NormSquared(operator.range).translated(data)

# Isotropic TV-regularization i.e. the l1-norm
l1_norm = 0.26 * odl.solvers.GroupL1Norm(gradient.range)

# Combine functionals, order must correspond to the operator K
f = odl.solvers.SeparableSum(l2_norm, l1_norm)


# --- Select solver parameters and solve using Chambolle-Pock --- #


# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.1 * odl.power_method_opnorm(op)

niter = 1000  # Number of iterations
tau = 0.1  # Step size for the primal variable
sigma = 1.0 / (op_norm ** 2 * tau)  # Step size for the dual variable
gamma = 0.1

# Optionally pass callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackPrint(lambda x: psnr(phantom.asarray(), x.asarray(), dynamic_range=np.max(phantom) - np.min(phantom))))

# Choose a starting point
x = pseudoinverse(data)

with odl.util.Timer('runtime of iterative algorithm'):
    # Run the algorithm
    odl.solvers.pdhg(
        x, f, g, op, tau=tau, sigma=sigma, niter=niter, gamma=gamma,
        callback=None)

print('ssim = {}'.format(ssim(phantom.asarray(), x.asarray())))
print('psnr = {}'.format(psnr(phantom.asarray(), x.asarray(), dynamic_range=np.max(phantom) - np.min(phantom))))

# Display images
figure_folder = ''
x.show('', clim=[0, 1], saveto=figure_folder + 'shepp_logan_tv')
x.show('', clim=[0.1, 0.4], saveto=figure_folder + 'shepp_logan_tv_windowed')