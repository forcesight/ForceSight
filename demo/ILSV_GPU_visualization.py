#!/usr/bin/env python

'''
example to show optical flow

USAGE: video_correlation.py [<video_source>]

Keys:
    ESC    - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
from cv2 import cuda
import EasyPySpin
# from skimage.metrics import structural_similarity
# from sklearn.cluster import KMeans
from collections import Counter
# from skimage.filters import sobel
import os, os.path
import re
import random
# from PIL import Image 
# from scipy.io import loadmat
import pandas as pd
# import matplotlib.pyplot as plt
from pathlib import Path
from numpy import linalg as LA
from tqdm import tqdm
#transforms
# from skimage import color
# from skimage.feature import match_template
import fnmatch
import queue
import time
# from scipy import zeros, signal, random

np.random.seed(42)


def nothing(x):
    pass

def draw_flow(img, flow,yy,xx):
    h, w = img.shape[:2]
    y, x = flow.shape[:2] 
    # print('hw',h,w)
    # print('yx',y,x)
    mask = np.zeros((h,w,3), np.uint8)
    # print(flow[10][10])
    scalar = int(20)
    for iy in range(y):
        for ix in range(x):
            start_point = (xx[ix],yy[iy])
            end_point = (xx[ix]+scalar*int(flow[iy][ix][0]),yy[iy]+scalar*int(flow[iy][ix][1]))
            # print(start_point,end_point)
            cv.arrowedLine(mask, start_point,end_point, (0, 255, 0),5)
    added_image = cv.addWeighted(img,1,mask,1,0)
    return added_image

cv.namedWindow('Touch Detection',cv.WINDOW_NORMAL)
cv.namedWindow('Original',cv.WINDOW_NORMAL)
cv.createTrackbar('Scale','Touch Detection',20,50,nothing)
cv.createTrackbar('Shift','Touch Detection',94,200,nothing)
# cv.resizeWindow('Touch Detection', 768,1024)
# cv.resizeWindow('Original', 768,1024)

def main():
    import sys
    try:
        fn = sys.argv[1]
        cam = cv.VideoCapture(fn)
        mode = 'video'
    except IndexError:
        fn = 0
        mode = 'camera'
        # cam = EasyPySpin.VideoCapture(0)
        cam = cv.VideoCapture(0)

    _ret, prev = cam.read()
    prev = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    gpu_previous = cv.cuda_GpuMat()
    gpu_previous.upload(prev)
    
    interval = 1

    
    while (cam.isOpened() == False):
        _ret, prev = cam.read()
        prev = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
        gpu_previous.uplaod(prev)
    
    # frameCount = int(cam.get(cv.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
    gpu_img = cv.cuda_GpuMat()
    gpu_ref = cv.cuda_GpuMat()
    gpu_ref = gpu_previous

    count = 0
    no_row_grids = 15#30
    no_col_grids = 30#60

    
    # flow_val = []
    nc = frameWidth # 2048
    nr = frameHeight # 1536
    length = int(cam.get(cv.CAP_PROP_FRAME_COUNT))
    vel_est_delta = 4 #delta of frames over which velocity is estimated
    integration = np.zeros([nr,nc,2])

    yy = np.arange(0,nr,int(nr/no_row_grids))
    xx = np.arange(0,nc,int(nc/no_col_grids))
    # print('xx,yy',len(xx),len(yy))

    queue_vel = queue.Queue(vel_est_delta)

    while (cam.isOpened() and count < length):
        # cam.set(cv.CAP_PROP_POS_FRAMES,count) # comment it if (mode == 'camera'):
        _ret, img = cam.read()
        gpu_img.upload(img)
        gpu_cur = cv.cuda.cvtColor(gpu_img, cv.COLOR_BGR2GRAY)
        # print(queue_vel.full())
        if (queue_vel.full() == False):
            queue_vel.put(gpu_cur)            
        else:
            gpu_ref = queue_vel.get()
            queue_vel.put(gpu_cur)
        try:
            start_calc_time = time.time()

            gpu_flow_create = cv.cuda_FarnebackOpticalFlow.create(3, 0.5, False, 15, 3, 5, 1.2, 0,)
            # print(gpu_ref.size(),gpu_cur.size())
            gpu_flow = cv.cuda_FarnebackOpticalFlow.calc(gpu_flow_create,gpu_ref,gpu_cur,None,)
            # flow = cv.calcOpticalFlowFarneback(rgb2gray(ref_img), rgb2gray(cur_img),None, 0.5, 3, 15, 3, 5, 1.2, 0)
            end_calc_time = time.time()
            print('calculation time per frame',end_calc_time-start_calc_time)

        except IndexError:
            continue
        start_draw_time = time.time()
        flow = gpu_flow.download()
        integration = np.add(integration, flow)
        flow_sparse = integration[::int(nr/no_row_grids),::int(nc/no_col_grids),:]
        # print('flow',flow.shape)
        # flow_val.append(flow_sparse)
        # flow = np.concatenate((x_list[count].reshape(len(yy),len(xx),1),y_list[count].reshape(len(yy),len(xx),1)),axis = 2)
        # print("flow",flow.shape)
        # print("flow_val",len(flow_val[0]))

        count = count + interval
        # gpu_ref = gpu_cur
        cv.imshow('Original',img)
        if count>=2*vel_est_delta:
            # flow_avg = np.mean(np.array(flow_val)[count-8:count,:,:,:],axis=0)
            # print('flow-avg_size',len(flow_avg))
            cv.imshow('Touch Detection',draw_flow(img,flow_sparse,yy,xx))
            # print('pass')
        # cv.imshow('flow', draw_flow(gray, flow))
        # cv.imshow('diff',draw_diff(gray,diff))
        end_draw_time = time.time()
        print('drawing time per frame',end_draw_time-start_draw_time)
        print('frame count:',count)
        # print('diff between prev frame and this frame:',diff)

        ch = cv.waitKey(5)
        if ch == 27:
            break
        if ch == ord('p'):
            cv.waitKey(-1)#pause
        if ch == ord('z'):
            integration = np.zeros([nr,nc,2])

    print('Done. frame count:',count)


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
