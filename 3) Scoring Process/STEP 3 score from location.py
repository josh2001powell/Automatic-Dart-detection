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


    #cv2.imshow('contrours + circle', contours_drawn)
    #cv2.imwrite('contours plus tip loc.jpg', contours_drawn)
    #cv2.waitKey(-1)


    return(rightmost_point)

def LOC2SCORE(dart_loc):

    centre = (400, 400)

    x = dart_loc[0] - centre[0]        ## x displacement from centre
    y = dart_loc[1] - centre[1]        ## y displacement from centre
    radius = np.sqrt(x**2 + y**2)      ## total distance from centre

    if x == 0:
        if y <= 0:
            ang_from_cent = 0
        elif y > 0:
            ang_from_cent = 180
        
    if x>0 and y<0:
        ang_from_cent = np.rad2deg(np.arcsin(np.abs(x)/radius))
    elif x>0 and y>0:
        ang_from_cent = np.rad2deg(np.arccos(np.abs(x)/radius)) + 90
    elif x<0 and y>0:
        ang_from_cent = np.rad2deg(np.arcsin(np.abs(x)/radius)) + 180
    elif x<0 and y<0:
        ang_from_cent = np.rad2deg(np.arccos(np.abs(x)/radius)) + 270


    ###############################################################################################################################################################
    ##############################################################  Turn angle and radius into score  #############################################################

    dartboard_scores = [1,18,4,13,6,10,15,2,17,3,19,7,16,8,11,14,9,12,5]
    lower = 9
    upper = 27

    if ang_from_cent > 353 or ang_from_cent < 27:
        dart_score = 20 

    for score in dartboard_scores:
        if ang_from_cent >= lower and ang_from_cent < upper:
            dart_score = score
        lower = upper 
        upper += 18

    if radius > 340:
        dart_score = 0
    elif radius <= 14:
        dart_score = 50
    elif radius <= 32 and radius > 14:
        dart_score = 25
    elif radius <= 214 and radius > 194:
        dart_score = 3 * dart_score 
    elif radius <= 340 and radius > 320:
        dart_score = 2 * dart_score   

    return(dart_score)


def dart_transform(dart_orig_loc, transformations):
    dart_temp_loc = dart_orig_loc
    for matrix in transformations:

        p = dart_temp_loc   ## easier to write just 'p' 
    
        px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))   # how x coordinate transforms
        py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))   # how y coordinate transforms
        
        dart_temp_loc = [px, py]

    dart_final_loc = [px, py]
    return(dart_final_loc)



tip = (370,200)

calibration_data = np.load(r'C:\Users\joshi\Documents\test folder\calibration_data.npy')
calibration_pic = cv2.imread('calibration pic.jpg')

actual_tip = dart_transform(tip, calibration_data)   ## dart transform moves tip to idealised board

Score = LOC2SCORE(actual_tip)




text = str(Score)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 2
text_color = (0, 0, 255)  # White
bg_color = (255, 255, 255)  # Red
position = tip

(text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)     ## size of text


top_left = (position[0] + 10, position[1] - text_height + 5) 
bottom_right = (position[0] + text_width + 10, position[1] + baseline + 5)


cv2.rectangle(calibration_pic, top_left, bottom_right, bg_color, thickness=cv2.FILLED)
cv2.putText(calibration_pic, str(Score), (tip[0]+10, tip[1]+10), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), thickness=4)


cv2.circle(calibration_pic, tip, 5, (0, 0, 255), -1)
cv2.imshow('before', calibration_pic)
cv2.waitKey(-1)

cv2.imwrite('dart location with score.jpg', calibration_pic)
