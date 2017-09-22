"""Reference FBP reconstruction for ellipse data."""

import numpy as np
import odl
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr


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


with odl.util.Timer('runtime of FBP reconstruction'):
    recon = odl.tomo.fbp_op(operator, filter_type='Hann')(data)

print('ssim = {}'.format(ssim(phantom.asarray(), recon.asarray())))
print('psnr = {}'.format(psnr(phantom.asarray(), recon.asarray(), dynamic_range=1)))

# Show without filter
initial = odl.tomo.fbp_op(operator)(data)

figure_folder = ''
phantom.show('', clim=[0, 1], saveto=figure_folder + 'shepp_logan_phantom')
phantom.show('', clim=[0.1, 0.4], saveto=figure_folder + 'shepp_logan_phantom_windowed')
data.show('', saveto=figure_folder + 'shepp_logan_data')
initial.show('', clim=[0, 1], saveto=figure_folder + 'shepp_logan_initial')
recon.show('', clim=[0, 1], saveto=figure_folder + 'shepp_logan_fbp')
recon.show('', clim=[0.1, 0.4], saveto=figure_folder + 'shepp_logan_fbp_windowed')
