import numpy as np
import argparse
import cv2
import os
def max_rgb_filter(image):
    (B, G, R) = cv2.split(image)
    M = np.maximum(np.maximum(R, G), B)
    R[R < M] = 0
    G[G < M] = 0
    B[B < M] = 0
    return cv2.merge([B, G, R])
input_folder = "D:/Research-CADS/Datasets/Fish/60FPS/F4"
output_folder = "D:\Research-CADS\Image Processing\Max-RGB\Result\F4"
for filename in os.listdir(input_folder):
     img = cv2.imread(os.path.join(input_folder, filename))
     filtered = max_rgb_filter(img)
     cv2.imwrite(os.path.join(output_folder, filename), filtered)
