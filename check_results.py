# -*- coding: utf-8 -*-
"""check_results.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nOQV0Pzl2pgkS6q0Y7Kq6RUgC1HnVC5q
"""

import numpy as np
import matplotlib.pyplot as plt 
import cv2 
from math import cos, sin, atan
import os 
from google.colab.patches import cv2_imshow

def read_data(filename):
  output = []
  with open(filename) as f:
      for line in f: # read rest of lines
        output.append([float(x) for x in line.split()])
  output = np.array(output)
  return output

path_to_file_with_coordinates = "path.txt"
data = read_data(path_to_file_with_coordinates)
plt.plot(data[:,0], data[:,1]) # просмотр x и y координат