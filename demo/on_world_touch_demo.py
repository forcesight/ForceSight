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
# from cv2 import cuda
import EasyPySpin
#transforms
import queue
import mediapipe as mp
import sys

np.random.seed(42)

#global parameters
touch_threshold = 5 #vector magnitude. The lower, the more sensitive detection
centripetal_threshold = 2 #vector variance. The higher, the less possible hand motion is detected as speckle motion (false positive)
forceScale = int(5) #scalar. The bigger, the longer vectors look. does not affect flow
lineWidth = int(5) # line width of vectors.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
#configuration
    #laser torch + diffuser, on partition: touch_threshold = 3, centripetal_threshold = 1
    #laser projector + concave only, on partition:  touch_threshold = 3, centripetal_threshold = 1
    #laser projector + concave + diffuser, on partition:  touch_threshold = 3.5, centripetal_threshold = 2. can cover larger area, but SNR is lower -> var is large when stepping on floor.
#laser projector + concave only, on wall: active_threshold = 6, touch_threshold = 7, centripetal_threshold = 0.5

def draw_touching_point(wrist, index, width, height):
    blank_image = np.zeros((int(height),int(width),3), np.uint8)
    
    #drawing parameters
    white = (255,255,255)
    wrist_co = (int(wrist.x*width),int(wrist.y*height))
    # print('wrist',wrist_co)
    index_co = (int(index.x*width),int(index.y*height))
    coeff = 3
    ellipse_center = (int(coeff*wrist_co[0] + (1-coeff)*index_co[0]),
                    int(coeff*wrist_co[1] + (1-coeff)*index_co[1]))
    longeraxis = int(np.sqrt((index_co[0]-ellipse_center[0])**2
                    +(index_co[1]-ellipse_center[1])**2))
    axesLength = (longeraxis,30)
    angle = np.arctan2(wrist_co[1]-index_co[1],wrist_co[0]-index_co[0])
    angle = int(angle*180/np.pi)
    # print(angle)
    # # wrist
    # image = cv.circle(image, wrist_co, 5, (0,255,0), 5)
    # # index_fingertip
    # image = cv.circle(image, index_co, 5, (0,255,0), 5)
    #ellipse
    blank_image = cv.ellipse(blank_image, ellipse_center,axesLength,angle,0,360,white,-1)
    cropped_image = blank_image[97:394,119:515]
    dim = (2048,1536)
    resized = cv.resize(cropped_image, dim, interpolation = cv.INTER_AREA)
    resized = cv.cvtColor(resized,cv.COLOR_BGR2GRAY).astype("float32")

    return resized, index_co


def nothing(x):
    pass

def cart2pol(z):
    rho = np.sqrt(z[0]**2 + z[1]**2)
    phi = np.arctan2(z[1],z[0])
    return(rho, phi)

def touch_detection(flow,yy,xx,isTouched,img_mask):
    y, x = flow.shape[:2] 
    
    isEnter = False
    isExit = False

    flow_array_original = flow
    flow_array = []

    for iy in range(y):
        for ix in range(x):
            if img_mask[yy[iy],xx[ix]] == 0:
                mag, angle = cart2pol(flow_array_original[iy][ix])
                if mag > touch_threshold:
                    flow_array.append([mag,angle])
    
    if flow_array == []:
        flow_array.append([0,0])
    flow_array = np.array(flow_array)
    flow_array_var = np.var(flow_array[:,1])
    flow_array_magnitude = flow_array[:,0].mean()

    if flow_array_magnitude > touch_threshold and flow_array_var > centripetal_threshold:
        touch_text = "Speckle motion: touched" 
        if (isTouched == False):
            isEnter = True
        isTouched = True
    else:
        touch_text = "Speckle motion: no touch" 
        if (isTouched == True):
            isExit = True
        isTouched = False
        
    # touch_text = "var:"+str(round(flow_array_var,2))+" mag:"+str(round(flow_array_magnitude,2))
    return touch_text, isTouched, isEnter, isExit

def draw_flow(img, flow,yy,xx,touch_text,img_mask):
    h, w = img.shape[:2]
    y, x = flow.shape[:2] 
    mask = np.zeros((h,w,3), np.uint8)
   
    for iy in range(y):
        for ix in range(x):
            if img_mask[yy[iy],xx[ix]] == 0:
                start_point = (xx[ix],yy[iy])
                end_point = (xx[ix]+forceScale*int(flow[iy][ix][0]),yy[iy]+forceScale*int(flow[iy][ix][1]))
                # print(start_point,end_point)
                cv.arrowedLine(mask, start_point,end_point, (0, 255, 0),lineWidth)
    masked_image = cv.addWeighted(img,1,mask,1,0)
    
    #test display
    font = cv.FONT_HERSHEY_SIMPLEX
    # org
    org = (600, 100)
    # fontScale
    fontScale = 2
    # Blue color in BGR
    color = (0, 255, 0)
    # Line thickness of 2 px
    thickness = 5
    
    masked_image = cv.putText(masked_image, touch_text, org, font, 
                   fontScale, color, thickness, cv.LINE_AA)

    return masked_image

cv.namedWindow('Touch Detection',cv.WINDOW_NORMAL)


def main():
    try:
        fn = sys.argv[1]
        cam = cv.VideoCapture(fn)
        mode = 'video'
    except IndexError:
        fn = 0
        mode = 'camera'
        cam = EasyPySpin.VideoCapture(0)
        cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        

    _ret, prev = cam.read()
    gpu_previous = cv.cuda_GpuMat()
    gpu_previous.upload(prev)
    
    interval = 1

    while (cam.isOpened() == False):
        _ret, prev = cam.read()
        gpu_previous.upload(prev)
    
    frameWidth = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    print(width, height)
    gpu_cur = cv.cuda_GpuMat()
    gpu_ref = cv.cuda_GpuMat()
    gpu_ref = gpu_previous

    count = 0
    no_row_grids = 15#30
    no_col_grids = 30#60

    flow = np.zeros([no_row_grids,no_col_grids,2])
    # flow_val = []
    nc = frameWidth # 2048
    nr = frameHeight # 1536
    vel_est_delta = 1 #delta of frames over which velocity is estimated

    yy = np.arange(0,nr,int(nr/no_row_grids))
    xx = np.arange(0,nc,int(nc/no_col_grids))

    queue_vel = queue.Queue(vel_est_delta)

    intensity_mask = cv.imread('29/29_mask.png')
    intensity_mask = cv.cvtColor(intensity_mask,cv.COLOR_BGR2GRAY).astype("float32")
    img_mask = 255*intensity_mask
    img_mask = img_mask.clip(0, 255).astype("uint8")

    UI = np.zeros((480,640,3), np.uint8) #297,396
    UI.fill(50)
    isTouched = False
    isEnter = False
    isExit = False

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while (cam.isOpened()):
            _ret, img = cam.read()
            gpu_cur.upload(img)

            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

            # print(bool(results.multi_hand_landmarks))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:

                    # mp_drawing.draw_landmarks(
                    # image,
                    # hand_landmarks,
                    # mp_hands.HAND_CONNECTIONS,
                    # mp_drawing_styles.get_default_hand_landmarks_style(),
                    # mp_drawing_styles.get_default_hand_connections_style())

                    hand_mask, index_co = draw_touching_point(hand_landmarks.landmark[0],
                                                    hand_landmarks.landmark[8],width,height)
                    # print(index_co,'index_co')
                    #GREEN
                    # image = cv.circle(image, index_co, 5, (0,255,0), 5)
                    
                    #RED
                    if isEnter == True:
                        UI = cv.circle(UI, index_co, 10, (0,0,255), -1)
                        print('red',index_co)
                    # if isExit == True:
                    #     UI.fill(50)
                    img_mask = 255*(intensity_mask+hand_mask)
                    img_mask = img_mask.clip(0, 255).astype("uint8")
            
            cv.imshow('Mask', cv.resize(img_mask, (640,480), interpolation = cv.INTER_AREA))
            cv.imshow('UI', cv.resize(UI[97:394,119:515], (800,600), interpolation = cv.INTER_AREA))
            
        
            try:
                gpu_flow_create = cv.cuda_FarnebackOpticalFlow.create(3, 0.5, False, 15, 3, 5, 1.2, 0,)
                gpu_flow = cv.cuda_FarnebackOpticalFlow.calc(gpu_flow_create,gpu_ref,gpu_cur,None,)

            except IndexError:
                continue
            flow = gpu_flow.download()
            flow_sparse = flow[::int(nr/no_row_grids),::int(nc/no_col_grids),:]


            count = count + interval
            
            cv.imshow('Webcam',image)
            if count>=2*vel_est_delta:
                touch_text, isTouched, isEnter, isExit = touch_detection(flow_sparse,yy,xx,isTouched,img_mask)
                display_img = draw_flow(cv.cvtColor(img,cv.COLOR_GRAY2BGR),
                                        flow_sparse,yy,xx,touch_text,img_mask)
                cv.imshow('Touch Detection',display_img)

            
            gpu_ref.upload(img)

            ch = cv.waitKey(5)
            if ch == 27:
                break
            if ch == ord('p'):
                cv.waitKey(-1)#pause
            if ch == ord('z'):
                UI.fill(50)

        print('Done. frame count:',count)


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
