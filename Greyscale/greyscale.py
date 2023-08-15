import cv2
import numpy as np
import os
input_folder = "D:/Research-CADS/Datasets/Fish/60FPS/F4"
output_folder = "D:\Research-CADS\Image Processing\Greyscale\Result\F4"
for filename in os.listdir(input_folder):
    img = cv2.imread(os.path.join(input_folder, filename))
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_folder, filename), gray_image)