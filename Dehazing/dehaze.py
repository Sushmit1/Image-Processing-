import cv2
import numpy as np
import os

def dark_channel(img, window=15):
    # Create a copy of the image
    dc = np.zeros_like(img)

    # Compute the minimum value within a local window
    for i in range(3):
        dc[:,:,i] = cv2.erode(img[:,:,i], np.ones((window,window),np.uint8))

    return dc.min(axis=2)

def estimate_atmosphere(img, percent=0.1):
    # Calculate the number of pixels to consider for estimating the atmosphere
    num_pixels = int(percent * img.shape[0] * img.shape[1])

    # Get the indices corresponding to the top num_pixels brightest pixels
    bright_indices = np.unravel_index(np.argsort(img.ravel())[::-1][:num_pixels], img.shape)

    # Estimate the atmosphere as the average pixel values of the bright pixels in the original image
    atmosphere = np.zeros(3)
    for i in range(3):
        atmosphere[i] = np.mean(img[bright_indices[0],bright_indices[1],i])

    return atmosphere

def dehaze(img, w=0.95, t0=0.01, window=15, percent=0.1):
    # Convert the image to floating point type
    img = img.astype('float32') / 255.0

    # Apply the dark channel prior
    dc = dark_channel(img, window)

    # Estimate the atmosphere
    atmosphere = estimate_atmosphere(img, percent)

    # Compute the transmission map
    t = 1 - w * dark_channel(img / atmosphere, window)

    # Clip the transmission map to avoid division by zero
    t = np.clip(t, t0, 1)

    # Perform the dehazing operation
    result = np.zeros_like(img)
    for i in range(3):
        result[:,:,i] = (img[:,:,i] - atmosphere[i]) / t + atmosphere[i]

    # Clip the result to the valid range
    result = np.clip(result, 0, 1)

    # Convert the result back to uint8 type
    result = (result * 255).astype('uint8')

    return result
    
# Process all images in a folder
input_folder = "C:/Users/sushm/Downloads/Haze.jpg"
output_folder = "D:/Research-CADS"

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the image
        img = cv2.imread(os.path.join(input_folder, filename))

        # Apply the dehazing algorithm
        result = dehaze(img)

        # Save the result image
        cv2.imwrite(os.path.join(output_folder, filename), result)
