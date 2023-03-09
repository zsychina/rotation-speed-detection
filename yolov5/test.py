import cv2 as cv
import numpy as np
import os

# print(os.path.exists('./video/walk.mp4'))
print(os.getcwd())
os.chdir('E:/innovProject/yolov5')
print(os.getcwd())

# vidcap = cv.VideoCapture('E:/innovProject/video/walk.mp4')
vidcap = cv.VideoCapture('../video/walk.mp4')

success, image = vidcap.read()
print(success)
# print(image)
