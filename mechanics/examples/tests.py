from pathlib import Path
import numpy as np
from mechanics.src.utils import morozov, load_images_and_displacements, load_order_clean, generate_mask_on_micro_image, remap
import matplotlib.pyplot as plt

# noise_path = Path("data/noise_experiment_T_100.0_E_1000.0_nu_0.3/img_1")
# images, displacements = load_images_and_displacements(noise_path, mode="noisy")

# plt.imshow(images[6][1])
# plt.show()

# image = images[6]

# homogeneous_patches = np.concatenate([
# image[0][25:50, 0:25].ravel(),
# image[0][250:275, 250:275].ravel(),
# image[0][250:275, 150:175].ravel()
# ])

# plt.imshow(image[0][250:275, 150:175])
# plt.show()


# micro_path = "/Users/josephinelahmani/Desktop/images_matthieu/260225/WT_gels1p2/250225_WT_1p2CD100RGD_FastAcq_pos1.tif" #"/Users/josephinelahmani/Desktop/250213_Overnight 50Hz 25Vpp_pos2.tif"
# img, _ = load_order_clean(micro_path)

# image = img[1:3,9,100:400,100:400,2]

# plt.imshow(image[0])
# plt.show()

# homogeneous_patches = np.concatenate([
# image[0][25:50, 25:50].ravel(),
# image[0][250:275, 250:275].ravel(),
# image[0][200:225, 200:225].ravel()
# ])

# plt.imshow(image[0][200:225, 200:225])
# plt.show()

# u, alpha, beta = morozov(image, num_iter_of=100, num_warp_of= 5, num_pyramid_of=10, pyramid_downscale_of=1.5, homogeneous_patches=homogeneous_patches, alpha_init=1.0, c=0.1, step_size=0.1)

# print(alpha, beta)


micro_path = "/Users/josephinelahmani/Desktop/250213_Overnight 50Hz 25Vpp_pos2.tif"
img, _ = load_order_clean(micro_path)
image = img[2:4, 10, 180:430, 140:390, 1] 

vmin = 0.48
vmax = 0.58
alpha = 0.3
c0 = remap(image[0], vmin, vmax)
c1 = remap(image[1], vmin, vmax)

newim = np.ones(image[0].shape + (3,))

newim[..., 0] = c0
newim[..., 1] = c1
newim[..., 2] = (1-alpha)*c0 + alpha*c1#*=remap(0.52, 0.45, 0.6)

fig, axes = plt.subplots(1, 3, figsize=(20, 4))

im0 = axes[0].imshow(newim)
axes[0].set_title("rgb")

im1 = axes[1].imshow(image[0], cmap='gray')
axes[1].set_title("r")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(image[1], cmap='gray')
axes[2].set_title("g")
plt.colorbar(im2, ax=axes[2])
plt.show()


# plt.imshow(newim)
# plt.colorbar()
# plt.show()