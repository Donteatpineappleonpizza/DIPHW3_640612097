from skimage import io, color, morphology, measure, filters
import numpy as np
import matplotlib.pyplot as plt

# Load images
image_1h = io.imread('WormHole_1H.tif')
image_2h = io.imread('WormHole_2H.tif')

# Define the hole counting function using previously defined logic
def count_holes(image):
    # Convert to grayscale
    gray_image = color.rgb2gray(image)

    # Invert the image to highlight the holes
    inverted_image = np.max(gray_image) - gray_image

    # Threshold the image to isolate the holes
    thresh = filters.threshold_otsu(inverted_image)
    binary = inverted_image > thresh

    # Perform morphological operations to clean up the image
    opened = morphology.binary_opening(binary, morphology.disk(3))
    closed = morphology.binary_closing(opened, morphology.disk(1))

    # Label the image to find connected regions
    labeled = measure.label(closed)
    properties = measure.regionprops(labeled)

    # Define a method to calculate circularity
    def circularity(region):
        return (4 * np.pi * region.area) / (region.perimeter ** 2)

    # Define criteria for what you consider a hole
    def is_hole(region):
        return (circularity(region) > 0.8 and region.area >= 100)

    # Filter regions based on the defined criteria
    holes = [prop for prop in properties if is_hole(prop)]
    
    # Return the number of holes
    return len(holes), holes

# Process images and count holes
num_holes_1h, holes_1h = count_holes(image_1h)
num_holes_2h, holes_2h = count_holes(image_2h)

# Plot the results for both images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Display image 1H with marked holes
ax[0].imshow(image_1h, cmap='gray')
for prop in holes_1h:
    y0, x0 = prop.centroid
    ax[0].plot(x0, y0, '.r', markersize=15)
ax[0].set_title(f'Image WormHole_1H: {num_holes_1h} Holes')
ax[0].axis('off')

# Display image 2H with marked holes
ax[1].imshow(image_2h, cmap='gray')
for prop in holes_2h:
    y0, x0 = prop.centroid
    ax[1].plot(x0, y0, '.r', markersize=15)
ax[1].set_title(f'Image WormHole_2H: {num_holes_2h} Holes')
ax[1].axis('off')

plt.tight_layout()
plt.show()

(num_holes_1h, num_holes_2h)