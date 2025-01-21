import numpy as np
import pylab as pl

def regb2gray(rgb):
    """Convert an RGB image to grayscale."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

# Load and process the image
image_path = 'C:/Users/dolap/Downloads/Sierpinski_triangle.png'
rgb_image = pl.imread(image_path)
image = regb2gray(rgb_image)

# Find non-zero pixels
pixels = np.argwhere(image > 0)

Lx, Ly = image.shape[1], image.shape[0]
print(Lx, Ly)
print(pixels.shape)

# Compute the fractal dimension
scales = np.logspace(0.01, 1, num=10, endpoint=False, base=2)
Ns = []

for scale in scales:
    print("======= Scale:", scale)
    bins_x = np.arange(0, Lx, scale)
    bins_y = np.arange(0, Ly, scale)
    H, edges = np.histogramdd(pixels, bins=(bins_x, bins_y))
    Ns.append(np.sum(H > 0))

# Linear fit
coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)

# Plotting
pl.plot(np.log(scales), np.log(Ns), 'o', mfc='none')
pl.plot(np.log(scales), np.polyval(coeffs, np.log(scales)))
pl.xlabel('log $\epsilon$')
pl.ylabel('log N')
pl.savefig('fractal.png')

# Output results
fractal_dimension = -coeffs[0]
print("The Hausdorff dimension is", fractal_dimension)
np.savetxt('fractal_dimension_scaling.txt', np.array([fractal_dimension]))