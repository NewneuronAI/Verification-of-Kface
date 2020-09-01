import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
import csv
import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


img = cv2.imread("C:/Users/bitcamp/Downloads/Low_Resolution/19101513/S002/L25/E02/C7.jpg", cv2.IMREAD_COLOR)
img = img[15:,35:135]
cv2.imwrite('./testimage12.jpg',img)

# img = cv2.imread("C:/Users/bitcamp/Downloads/Low_Resolution/19082721/S001/L25/E01/C7.jpg",)
# cv2.imwrite('./testimage1.jpg',img)