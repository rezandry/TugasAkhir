import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from skimage import io

im = io.imread('datasetpaper.jpg')

plt.imshow(im, cmap='gray')

from skimage import restoration
from skimage import img_as_float
im_float = img_as_float(im)

im_denoised = restoration.nl_means_denoising(im_float, h=0.05)

plt.imshow(im_denoised, cmap='gray')
ax = plt.axis('off')

plt.imshow(im_denoised, cmap='gray')
click_markers = plt.ginput(n=-1, timeout=-1)

x, y = np.load('indices_markers.npy').T
plt.imshow(im_denoised, cmap='gray')
plt.plot(y, x, 'or', ms=4)

from skimage import morphology
markers = np.zeros(im_denoised.shape, dtype=np.int)
markers[x.astype(np.int), y.astype(np.int)] = np.arange(len(x)) + 1
markers = morphology.dilation(markers, morphology.disk(7))

from scipy import ndimage
from skimage import morphology
# Black tophat transformation (see https://en.wikipedia.org/wiki/Top-hat_transform)
hat = ndimage.black_tophat(im_denoised, 7)
# Combine with denoised image
hat -= 0.3 * im_denoised
# Morphological dilation to try to remove some holes in hat image
hat = morphology.dilation(hat)
plt.imshow(hat, cmap='spectral')

labels_hat = morphology.watershed(hat, markers)
from skimage import color
color_labels = color.label2rgb(labels_hat, im_denoised)
plt.imshow(color_labels[:300, :300])

# A different markers image: laplace filter
lap = ndimage.gaussian_laplace(im_denoised, sigma=0.7)
lap = restoration.nl_means_denoising(lap, h=0.002)
plt.imshow(lap, cmap='spectral'); plt.colorbar()

labels_lap = morphology.watershed(lap, markers)
color_labels = color.label2rgb(labels_lap, im_denoised)
plt.imshow(color_labels)