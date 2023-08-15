import cv2
import numpy as np
import os

input_folder = "D:/Research-CADS/Datasets/Fish/60FPS/F4"
output_folder = "D:\Research-CADS\Image Processing\HE\Result\F4"
for filename in os.listdir(input_folder):
    img = cv2.imread(os.path.join(input_folder, filename),0)
    equ = cv2.equalizeHist(img)
    cv2.imwrite(os.path.join(output_folder, filename), equ)









