import cv2
import numpy as np
import os

input_folder = "D:/Research-CADS/Datasets/Fish/60FPS/F4"
output_folder = "D:\Research-CADS\Image Processing\CLAHE\Result\F4"
for filename in os.listdir(input_folder):
    image = cv2.imread(os.path.join(input_folder, filename))
    image = cv2.resize(image, (500, 600))
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5)
    final_img = clahe.apply(image_bw) + 30
    cv2.imwrite(os.path.join(output_folder, filename), final_img)
