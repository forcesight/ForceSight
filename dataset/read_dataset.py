#!/usr/bin/env python

'''
example to show optical flow

USAGE: evaluation_GPU.py [<video_source>]

Keys:
    ESC    - exit
'''
import h5py #need to be put in front of cv2 (I dont know why)
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#transforms

import queue
import time
import os
import fnmatch
import csv


np.random.seed(42)



def read_csv(force_name):
    l = []
    with open(force_name, newline='') as csvfile: #CSV file location
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            if len(row)>0:
                l.append(row[1])
                
    l_new = []
    for l_el in l:
        try:
            l_new.append(float(l_el))
        except ValueError:
            l_new.append(l_new[-1])
    return np.abs(np.array(l_new))

def access_data(i,j):
    ## Exp No. i
    No = str(i)
    ## Exp trial. 0 is no action, 1-5 are repeating trials, each contains one ramp.
    trial = str(j)
    path = No +'/'
    fn = No +'_'+trial+'*.avi'
    for file in os.listdir(path):
        if fnmatch.fnmatch(file, fn):
            filename = file
    print(filename)
    vid_name = path+filename
    print(vid_name)
    mask_filename = No +'_mask.png'
    mask_name = path+mask_filename
    print(mask_name)
    force_filename = 'data_'+ No+'_'+trial+'.csv'
    force_name = path + force_filename
    print(force_name)

    with open(path+'gt.txt','r') as f:
        _, gt_x, gt_y = f.readline().strip().split(',')
        gt_x = int(gt_x)
        gt_y = int(gt_y)
    return vid_name, mask_name, force_name, gt_x, gt_y


def nothing(x):
    pass

def draw_flow(img, flow,yy,xx):
    h, w = img.shape[:2]
    r, c = flow.shape[:2] 
    # print('hw',h,w)
    # print('yx',y,x)
    mask = np.zeros((h,w,3), np.uint8)
    # print(flow[10][10])
    for iy in range(r):
        for ix in range(c):
            start_point = (xx[ix],yy[iy])
            end_point = (int(xx[ix]+flow[iy][ix][0]),int(yy[iy]+flow[iy][ix][1]))
            # print(start_point,end_point)
            cv.arrowedLine(mask, start_point,end_point, (0, 255, 0),5)
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    added_image = cv.addWeighted(img,1,mask,1,0)
    return added_image


def cal_velocity(flow_per_frame,dir_x,dir_y): # flow_per_frame is (384, 512, 2)
    # sign = np.sign(x*dir_x + y*dir_y)
    flow_x = flow_per_frame[:,:,0]
    flow_y = flow_per_frame[:,:,1]
    velocity = flow_x*dir_x + flow_y*dir_y
    return velocity


# cv.namedWindow('Velocity',cv.WINDOW_NORMAL)
# cv.namedWindow('Original',cv.WINDOW_NORMAL)



def main():
    # No
    for i in range(1,22): #range(a,b) from a to b-1
        if i == 15 or i==16:
            continue
        No = str(i)
        f = h5py.File('./output/'+No+'_sam_test.h5', 'r')
        # trial
        for j in range(1,2): # 1,2,3,4,5
            trial = str(j)
            vid_name, mask_name, force_name, gt_x, gt_y = access_data(i,j)
            dataset = f.get(trial+'_trial')

            nr = 1536
            nc = 2048

            no_row_grids = 384 # H
            no_col_grids = 512 # W
            frameCount = int(len(dataset)/no_row_grids)
            print(frameCount) # N 

            # convert "dataset" into 4D array "flow" N X H X W X 2. It takes ~30s
            flow = dataset[:frameCount*384*12*2].reshape(frameCount,384,512,2)
            print(flow.shape)

            # yy = np.arange(0,nr,int(nr/no_row_grids))
            # xx = np.arange(0,nc,int(nc/no_col_grids))
            # X,Y = np.meshgrid(xx,yy)
            # dir_x = gt_x - X
            # dir_y = gt_y - Y
            # dir_x = dir_x/(np.sqrt(dir_x**2+dir_y**2))
            # dir_y = dir_y/(np.sqrt(dir_x**2+dir_y**2))
            # print('xx,yy',len(xx),len(yy))
            
            # dist_list = np.cumsum(np.array(vel_avg_list))           
            # force_list = read_csv(force_name)
            # fig, ax = plt.subplots(3,1)
            # ax[0].set_title('velocity')
            # ax[0].plot(range(len(vel_avg_list)),vel_avg_list)
            # ax[0].set_xlim([0,frameCount])
            # ax[1].set_title('distance')
            # ax[1].plot(range(len(vel_avg_list)),dist_list)
            # ax[1].set_xlim([0,frameCount])
            # ax[2].set_title('Force curve gauge')
            # ax[2].plot(range(int(len(vel_avg_list)/10)),force_list[0:int(len(vel_avg_list)/10)])
            # ax[2].set_xlim([0,int(len(force_list))])
            # plt.show()
        f.close()


    print('Done. frame count:',count)


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
