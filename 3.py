import imageio
import cv2
import numpy as np


def custom_median_filter(image, window_size):
  """Implements a custom median filter on a grayscale image.

  Args:
      image: A numpy array representing the grayscale image.
      window_size: The size of the median filter window (must be odd).

  Returns:
      A numpy array representing the filtered image.
  """
  filtered_image = np.zeros_like(image)
  height, width = image.shape

  for y in range(height):
    for x in range(width):
      window_y_min = max(0, y - window_size // 2)
      window_y_max = min(height, y + window_size // 2 + 1)
      window_x_min = max(0, x - window_size // 2)
      window_x_max = min(width, x + window_size // 2 + 1)

      window = image[window_y_min:window_y_max, window_x_min:window_x_max]
      filtered_image[y, x] = np.median(window)

  return filtered_image


def segment_and_count_holes(image):
  """Segments the eggplant from the background and counts holes (dark regions).

  Args:
      image: A numpy array representing the TIF image.

  Returns:
      A tuple containing the segmented eggplant image and the number of holes.
  """
  # Convert the image to grayscale
  grayscale_image = image.mean(axis=2)

  # Apply thresholding to segment the eggplant
  threshold = np.mean(grayscale_image) / 2
  segmented_image = (grayscale_image > threshold).astype(np.uint8) * 255

  # Apply custom median filter to remove noise
  filtered_image = custom_median_filter(segmented_image, 20)

  # Count the number of holes (connected black regions)
  _, labels, stats, _ = cv2.connectedComponentsWithStats(filtered_image, connectivity=4)
  holes = np.sum(stats[1:, cv2.CC_STAT_AREA] > 20)  # Adjust area threshold as needed

  return grayscale_image, segmented_image, holes


# Read the TIF image
image = imageio.imread('WormHole_1H.tif')

grayscale_image, segmented_image, hole_count = segment_and_count_holes(image.copy())

# Refine segmentation and hole detection (adjust values as needed)
more_selective_threshold = 0.3 * np.mean(grayscale_image)
refined_segmented_image = (grayscale_image > more_selective_threshold).astype(np.uint8) * 255
filtered_image = custom_median_filter(refined_segmented_image, 30)  # Slightly larger window
_, labels, stats, _ = cv2.connectedComponentsWithStats(filtered_image, connectivity=4)
holes = np.sum(stats[1:, cv2.CC_STAT_AREA] > 25)  # Increased area threshold

print(f"Number of refined holes detected: {holes}")
# Visualize results (optional)
cv2.imshow('Original Image', image)
cv2.imshow('Refined Segmented Image', refined_segmented_image)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)

