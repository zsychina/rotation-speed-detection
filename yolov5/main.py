import numpy as np
import cv2 as cv
import torch
import pandas as pd
import time
from Kalmantool import KalmanFilter
from Analyze import LimitedQueue
from Analyze import CalPeriod
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.linear_model import LinearRegression

def linear_regression_intercept(data):
    # 将原始数据转换为NumPy数组，并将x轴坐标设为该数组的索引值
    data = np.array(data)
    x = np.arange(len(data)).reshape(-1, 1)

    # 使用Z-score方法识别离群点，去除离群点后的数据集为filtered_entries
    z_scores = stats.zscore(data)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3)
    filtered_data = data[filtered_entries]
    x = x[filtered_entries]

    # 创建并训练线性回归模型
    model = LinearRegression()
    model.fit(x, filtered_data)

    # 返回线性回归模型的纵截距
    return model.intercept_


def rotation_pred(recent_x):
    data_aug = 1
    flame_len = len(recent_x)
    y = np.array(range(flame_len))
    k, _ = np.polyfit(recent_x * data_aug, y, deg=1)
    print(k)
    if k < 0: return 1
    else: return 0

bullet_time = 0.1 # example bullet_time=0.1

rotation_dir = 1 # 1: to left; 0: to right

# meta params
QUEUE_LEN = 500

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s') # download from github
# model = torch.hub.load('ultralytics/yolov5', 'custom', 'runs/train/exp14/weights/best.pt')
model = torch.hub.load('ultralytics/yolov5', 'custom', 'runs/train/exp18/weights/best.pt')

# cap = cv.VideoCapture(0)
cap = cv.VideoCapture('E:/innovProject/video/RotationTarget.mp4')

# set Kalman
dt = 1.0/15
F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
H = np.array([1, 0, 0]).reshape(1, 3)
Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
R = np.array([0.5]).reshape(1, 1)
kf_x = KalmanFilter(F=F, H=H, Q=Q, R=R)
kf_y = KalmanFilter(F=F, H=H, Q=Q, R=R)

# locx_base = []
# locy_base = []
# predictions_x = []
# predictions_y = []
locx_base = LimitedQueue(QUEUE_LEN)
locy_base = LimitedQueue(QUEUE_LEN)
predictions_x = LimitedQueue(QUEUE_LEN)
predictions_y = LimitedQueue(QUEUE_LEN)

period_stat = []

current_fps = 0
if not cap.isOpened():
    print('Cannot open camera')
    exit()
while True:
    time_start = time.time()
    ret, image = cap.read()
    if not ret:
        print('Cannot receive frame')
        break
    
    current_fps += 1
    results = model(image)
    locations = results.pandas().xyxy[0]
    try:
        # get locations named 'mark'
        obj = locations.loc[locations['name'] == 'mark'] # target name

        # get first target locations
        obj_x = obj.iloc[0]['xmin']
        obj_y = obj.iloc[0]['ymin']
        obj_width = obj.iloc[0]['xmax'] - obj_x
        obj_height = obj.iloc[0]['ymax'] - obj_y
        obj_x = float(obj_x)
        obj_y = float(obj_y)

        # kalman update
        kf_x.update(obj_x)
        kf_y.update(obj_y)

        # after Kalman
        next_x = float(np.dot(H, kf_x.predict())[0])
        next_y = float(np.dot(H, kf_y.predict())[0])
        
        # visualize
        cv.rectangle(image, (int(obj_x), int(obj_y)), (int(obj_x+obj_width), int(obj_y+obj_height)), (0,255,0),2) # yolo
        cv.rectangle(image, (int(next_x), int(next_y)), (int(next_x+obj_width), int(next_y+obj_height)), (255,255,255),2) # kalman 
        

        if d_x != None:
            if rotation_dir == 1:
                cv.rectangle(image, (int(next_x-d_x), int(next_y)), (int(next_x+obj_width-d_x), int(next_y+obj_height)), (0,255,255),2)
            elif rotation_dir == 0:
                cv.rectangle(image, (int(next_x+d_x), int(next_y)), (int(next_x+obj_width+d_x), int(next_y+obj_height)), (0,255,255),2)


        # save data
        # - current location
        # locx_base.push(obj_x)
        # locy_base.push(obj_y)
        predictions_x.push(next_x)
        predictions_y.push(next_y)
        # - serialize
        x_arr = np.array(locx_base)
        y_arr = np.array(locy_base)

        x_pre_arr = np.array(predictions_x)
        y_pre_arr = np.array(predictions_y)



    except:
        # print('No obj detected')
        predictions_x.push(0)
        predictions_y.push(0)

        x_pre_arr = np.array(predictions_x)
        y_pre_arr = np.array(predictions_y)

    if current_fps > 10:
        current_period = CalPeriod.period(x_pre_arr, process_time)

        d_x = (1000/current_period) * bullet_time

        if len(x_pre_arr) > 200:
            try:
                rotation_dir = rotation_pred(x_pre_arr[-200:])
                # current_period = linear_regression_intercept(period_stat)
            except:
                pass

        
        # statistics
        period_stat.append(current_period)


        print('current_period: ', current_period)

    cv.imshow('obj', image)   
    time_end = time.time()
    process_time = time_end - time_start

    try: fps = 1 / process_time 
    except: fps = 0
    print('FPS: ', int(fps), 'serials: ', current_fps)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


# 实际上应该是1.81s左右

# import matplotlib
# matplotlib.use('TkAgg')

# plt.subplot(2, 1, 1)
# plt.plot(x_arr)
# plt.subplot(2, 1, 2)
# plt.plot(x_pre_arr)

# plt.show()

