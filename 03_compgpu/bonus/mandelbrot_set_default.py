import numpy as np
import matplotlib.pyplot as plt

# CONSTANTS: The x-min/max and the y-min/max used in the mandelbrot generation
X_MIN, X_MAX, Y_MIN, Y_MAX = -2, 1, -1, 1

def mandelbrot(c, max_iter=100):
    """Computes the number of iterations before divergence."""
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def mandelbrot_set(width, height, x_min, x_max, y_min, y_max, max_iter=100):
    """Generates the Mandelbrot set image."""
    x_vals = np.linspace(x_min, x_max, width)
    y_vals = np.linspace(y_min, y_max, height)
    image = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            c = complex(x_vals[j], y_vals[i])
            image[i, j] = mandelbrot(c, max_iter)

    return image

def generate_image_default(width, height):
    """
    Helper utility to generate the mandelbrot set with provided width and height.
    The x_min, x_max are -2, 1. The y_min, y_max are -1, 1. These values are defined
    as constants at the top of this file.
    Uses the default mandelbrot generation function (NOT OPTIMIZED). This form thus acts
    as a testing utility
    """
    return mandelbrot_set(width, height, X_MIN, X_MAX, Y_MIN, Y_MAX)

# ========================================================================
# NOTE: UNCOMMENT THIS CODE TO SEE THE MANDELBROT IMAGES GENERATED
#       THIS HAS BEEN COMMENTED TO MAKE SURE MATPLOTLIB DOESN'T INTERFERE
#       WITH PYTEST
# ========================================================================
# Parameters. The x and y min/max values are defined above. 
# width, height = 1000, 800

# # Generate fractal
# image = generate_image_default(width, height)

# # Display
# plt.imshow(image, cmap='inferno', extent=[X_MIN, X_MAX, Y_MIN, Y_MAX])
# plt.colorbar()
# plt.title("Mandelbrot Set (DEFAULT)")
# plt.show()
