#!usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression  # 导入先行回归模型
# 距离计算函数
def center_distance_to_camera(knownWidth, focalLength, WH_M):
    # compute and return the distance from the maker to the camera

    Predict_Distance = (knownWidth * focalLength) / WH_M

    return Predict_Distance

def get_distance(box):
    KNOWN_DISTANCE = 50.0  # cm
    TRUE_WIDTH = 28  # cm
    TRUE_HEIGHT = 28  # cm

    BOX_WIDTH = 210  # pix
    BOX_HEIGHT = 210  # pix

    xmin, ymin, xmax, ymax = box

    w_h_mean = ((xmax - xmin) + (ymax - ymin)) / 2
    focalLength = (BOX_WIDTH + BOX_HEIGHT) * KNOWN_DISTANCE / (TRUE_WIDTH + TRUE_HEIGHT)
    cam_distance = center_distance_to_camera((TRUE_WIDTH + TRUE_HEIGHT) / 2, focalLength, w_h_mean)

    x = np.array([[374 + 328.45], [224 + 198], [154 + 141], [110 + 100], [84 + 77]])
    y = np.array([30, 50, 70, 90, 120])
    regressor = LinearRegression(normalize=True).fit(x, y)
    regression_distance = regressor.predict((xmax - xmin) + (ymax - ymin))[0]
    distance = cam_distance if cam_distance < 80 else (regression_distance + cam_distance) * 0.5

    return distance
