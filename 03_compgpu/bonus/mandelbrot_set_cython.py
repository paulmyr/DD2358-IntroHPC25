import matplotlib.pyplot as plt
import cython_mandelbrot

# CONSTANTS: The x-min/max and the y-min/max used in the mandelbrot generation
X_MIN, X_MAX, Y_MIN, Y_MAX = -2, 1, -1, 1

def generate_image_cython(width, height):
    """
    Helper utility to generate the mandelbrot set with provided width and height.
    The x_min, x_max are -2, 1. The y_min, y_max are -1, 1. These values are defined
    as constants at the top of this file.
    Uses the default mandelbrot generation function (NOT OPTIMIZED). This form thus acts
    as a testing utility
    """
    return cython_mandelbrot.mandelbrot_set(width, height, X_MIN, X_MAX, Y_MIN, Y_MAX)

# ========================================================================
# NOTE: UNCOMMENT THIS CODE TO SEE THE MANDELBROT IMAGES GENERATED
#       THIS HAS BEEN COMMENTED TO MAKE SURE MATPLOTLIB DOESN'T INTERFERE
#       WITH PYTEST
# ========================================================================
# # Generate fractal
# # Parameters
# width, height = 2048, 2048
# image = generate_image_cython(width, height)

# # Display
# plt.imshow(image, cmap='inferno', extent=[X_MIN, X_MAX, Y_MIN, Y_MAX])
# plt.colorbar()
# plt.title("Mandelbrot Set (CYTHON)")
# plt.show()
