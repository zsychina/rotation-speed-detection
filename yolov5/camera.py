import numpy as np
import cv2 as cv
import torch
import pandas as pd
import time
from Kalmantool import KalmanFilter

model = torch.hub.load('ultralytics/yolov5', 'yolov5s') # download from github

cap = cv.VideoCapture(0)
# cap = cv.VideoCapture('../video/walk.mp4')

# set Kalman
dt = 1.0/60
F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
H = np.array([1, 0, 0]).reshape(1, 3)
Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
R = np.array([0.5]).reshape(1, 1)
kf_x = KalmanFilter(F=F, H=H, Q=Q, R=R)
kf_y = KalmanFilter(F=F, H=H, Q=Q, R=R)
locx_base = []
locy_base = []
predictions_x = []
predictions_y = []

if not cap.isOpened():
    print('Cannot open camera')
    exit()
while True:
    time_start = time.time()
    ret, image = cap.read()
    if not ret:
        print('Cannot receive frame')
        break

    results = model(image)
    locations = results.pandas().xyxy[0]
    try:
        # get locations
        obj = locations.loc[locations['name'] == 'person']
        obj_x = obj.iloc[0]['xmin']
        obj_y = obj.iloc[0]['ymin']
        obj_width = obj.iloc[0]['xmax'] - obj_x
        obj_height = obj.iloc[0]['ymax'] - obj_y
        obj_x = float(obj_x)
        obj_y = float(obj_y)
        kf_x.update(obj_x)
        kf_y.update(obj_y)
        # after Kalman
        next_x = float(np.dot(H, kf_x.predict())[0])
        next_y = float(np.dot(H, kf_y.predict())[0])
        # visualize
        cv.rectangle(image,(int(obj_x),int(obj_y)),(int(obj_x+obj_width),int(obj_y+obj_height)),(0,255,0),2)
        cv.rectangle(image,(int(next_x),int(next_y)),(int(next_x+obj_width),int(next_y+obj_height)),(255,255,255),2)


        # # save data
        # locx_base.append(obj_x)
        # locy_base.append(obj_y)
        # x_arr = np.array(locx_base)
        # y_arr = np.array(locy_base)
        # predictions_x.append(next_x)
        # predictions_y.append(next_y)

    except:
        print('Obj doesn\'t exist in this flame.')
    
    cv.imshow('obj',image)   




    time_end = time.time()
    process_time = time_end - time_start
    try:
        fps = 1 / process_time
    except:
        fps = 0
    print(fps)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
