#!/usr/bin/env python

'''
example to show optical flow

USAGE: video_correlation.py [<video_source>]

Keys:
    ESC    - exit
'''

# Python 2/3 compatibility
# from __future__ import print_function

from tracemalloc import start
import numpy as np
import cv2 as cv

import queue

np.random.seed(42)

def draw_flow(img, flow,yy,xx):
    h, w = img.shape[:2]
    y, x = flow.shape[:2] 
    mask = np.zeros((h,w,3), np.uint8)

    scalar = int(1)
    for iy in range(y):
        for ix in range(x):
            start_point = (xx[ix],yy[iy])
            end_point = (xx[ix]+scalar*int(flow[iy][ix][0]),yy[iy]+scalar*int(flow[iy][ix][1]))
            cv.arrowedLine(mask, start_point,end_point, (0, 255, 0), thickness = 2)
    added_image = cv.addWeighted(img,1,mask,1,0)

    return added_image

def find_angle(flow):

    scalar = int(3)

    flow_sparse_roi = scalar * flow[:, :, :]

    # Manually calibrate angles to lower computational burden
    thetas = [210, 180, 0]
    angs = []
    
    for theta in thetas:

        radianAngle = theta * np.pi / 180        
        xRef = 1 * np.cos(radianAngle)
        yRef = 1 * np.sin(radianAngle)

        dir_ref_map = np.zeros((15,31,2))
        dir_ref_map[:,:,0].fill(yRef)
        dir_ref_map[:,:,1].fill(xRef)

        ang_map = np.multiply(flow_sparse_roi, dir_ref_map)
        ang = np.sum(ang_map)
        angs.append(ang)

    angle = thetas[np.argmax(angs)]

    return angle

def button(flow, angle):

    scalar = int(1)
    flow_sparse_roi = scalar * flow[:, :, :]
    flow_mag_x = flow[:, :, 0] **2 
    flow_mag_y = flow[:, :, 1] **2 
    flow_mag = np.sqrt(flow_mag_x + flow_mag_y)
    flow_sparse_roi_sum = np.sum(flow_mag)

    if abs(flow_sparse_roi_sum) > 1200:
        # button pressed
        if angle == 210:
            button_status = "TILT LEFT"
        elif angle == 180:
            button_status = "TILT BACKWARD"
        elif angle == 0:
            button_status = "TILT FORWARD"
        # elif angle == 330:
        #     button_status = "LEFT"
    else:
        button_status = "NO TILT"
    
    return button_status

def rgb2gray(rgb):

    b, g, r = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def main():
    import sys
    try:
        fn = sys.argv[1]
        cam = cv.VideoCapture(0)
    except IndexError:
        fn = 0
        cam = cv.VideoCapture(0)

    _ret, prev = cam.read()
    
    while (cam.isOpened()== False):
        _ret, prev = cam.read()

    frameWidth = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))

    no_row_grids = 15#30
    no_col_grids = 30#60

    flow_val = []

    nc = frameWidth # 2048
    nr = frameHeight # 1536
    vel_est_delta = 1 #delta of frames over which velocity is estimated

    integration = np.zeros([480,640,2])

    yy = np.arange(0,nr,int(nr/no_row_grids))
    xx = np.arange(0,nc,int(nc/no_col_grids))

    queue_vel = queue.Queue(vel_est_delta)

    while (cam.isOpened()):
        _ret, img = cam.read()
        ref_img = prev
        cur_img = img

        if (queue_vel.full() == False):
            queue_vel.put(img)            
        else:
            ref_img = queue_vel.get()
            queue_vel.put(img)

        try:
            flow = cv.calcOpticalFlowFarneback(rgb2gray(ref_img), rgb2gray(cur_img),None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
        except IndexError:
            continue

        integration=np.add(integration,flow)
        flow_sparse = integration[::int(nr/no_row_grids),::int(nc/no_col_grids),:]

        angle = find_angle(flow_sparse)
        button_status = button(flow_sparse, angle)
        
        op_fl = draw_flow(img,flow_sparse,yy,xx)

        op_fl = cv.putText(op_fl, button_status, org = (100, 50), fontFace = cv.FONT_HERSHEY_DUPLEX,
                fontScale = 1.5, color = (0, 255, 0), thickness = 1)
        
        visualize = np.concatenate((img[:,100:580], op_fl[:,100:580]), axis=1)

        visualize_ = cv. resize(visualize,(3072, 1920)) # Resize image.
        cv.imshow('Raw Image', visualize_)

        ch = cv.waitKey(5)
        if ch == 27:
            break
        if ch == ord('p'):
            cv.waitKey(-1)
        if ch == ord('z'):
            integration = np.zeros([nr,nc,2])

if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
