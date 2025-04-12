import numpy as np
import queue
import threading
import tkinter as tk
import tkinter.ttk as ttk
import time
import math
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2


def get_dart_position(dart_image):
    dilate = dart_image.copy()
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    first = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel, iterations=1)     #####    removes little white dots that are noise
    final = cv2.dilate(first, kernel, iterations=1) 

    final = cv2.cvtColor(final, cv2.COLOR_RGB2GRAY)

    contours, hierarchy = cv2.findContours(final, 1, 1) 
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    centers_list = [] 

    contours_drawn = cv2.drawContours(dilate, contours, -1, (0, 255, 0), 2)



    ########################################## Finding center of largest contour ##########################################
    M = cv2.moments(contours_sorted[0])
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        largest_center = (cx, cy)                    ## center of largest contour      should be the flight of the dart
        
    ###########################################################################################################################################
    ## Now the goal is to ignore any other contour who's center is not close enough to the largest contour to be considered part of the dart ##
    ################################# https://www.geeksforgeeks.org/python-opencv-find-center-of-contour/  ####################################
    contour_index = 1
    removal_index_list = []
    

    ##################################### This loop is ignored if only 1 contour found #####################################
    for i in contours_sorted[1:]:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            center = (cx, cy)
            
            x_diff = cx - largest_center[0]
            y_diff = cy - largest_center[1]
            
            #print(x_diff, y_diff)
            dist_diff =  np.sqrt(x_diff**2 + y_diff**2)     ## how far is the current contours center from the largest contour

            dist_thresh = 70
            if dist_diff > dist_thresh:
                removal_index_list.append(contour_index)
            else:
                centers_list.append(center)
            
            contour_index += 1

    num_removed = 0
    for i in removal_index_list:
        contours_sorted.pop(i - num_removed)                   ## need this for wehen we remove multiple to not mess up the indexes
        num_removed += 1
    
    ###############################################################################################

    if len(contours_sorted) == 1:
        largest_contour = contours_sorted[0]
        
        rightmost_point = (0,0)
        for point in largest_contour:
            x = point[0][0]
            y = point[0][1]
            
            if x > rightmost_point[0]:
                rightmost_point = (x,y)
            
    ########################################  Finds the furthest right point from every acceptable contour  ########################################
    if len(contours_sorted) > 1:
        rightmost_point = (0,0)
        for cnt in contours_sorted:
            for point in cnt:
                x = point[0][0]
                y = point[0][1]
                if x > rightmost_point[0]:
                    rightmost_point = (x,y)   

    
    cv2.circle(contours_drawn, rightmost_point, 5, (0, 0, 255), -1)


    cv2.imshow('contrours + circle', contours_drawn)
    #cv2.imwrite('contours plus tip loc.jpg', contours_drawn)
    cv2.waitKey(-1)


    return(rightmost_point)



dart_image_dilated = cv2.imread(r'C:\Users\joshi\Documents\test folder\0000  dilate  2.jpg')
tip = get_dart_position(dart_image_dilated)


calibration_pic = cv2.imread('calibration pic.jpg')
cv2.circle(calibration_pic, tip, 5, (0, 0, 255), -1)
cv2.imshow('before', calibration_pic)



cv2.waitKey(-1)

