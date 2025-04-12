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


cam_feed_num = 1


##  This loads the camera instantly rather than delayed becasue of permissions etc


'''feed  = cv2.VideoCapture(0)

cv2.namedWindow("Main", cv2.WINDOW_NORMAL)  ## allows window to be made bigger



if not feed.isOpened():
    print('Cannot open camera')

while feed.isOpened:
    ret, frame = feed.read()
    #resized_frame = cv2.resize(frame, (1200, 900), interpolation=cv2.INTER_LINEAR)  
    cv2.imshow('Main', frame)
    
    
    if cv2.waitKey(1) == ord('q'):
        break

feed.release()
cv2.destroyAllWindows()
'''
###############################################################################################################################################
###############################################################################################################################################
##############################################    ALL FROM CALIBRATION FILE      ##############################################################
###############################################################################################################################################
###############################################################################################################################################
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
        dart_score = 3 * dart_score   


    return(dart_score)

clicked_point = None
def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"Clicked point: {clicked_point}")

def rotate_img(INP_IMG, anti_clock_rot_deg):
    
    h, w = np.shape(INP_IMG)[:2]
    center = (w/2, h/2)
    rotation_matrix = cv2.getRotationMatrix2D(center, anti_clock_rot_deg, 1)
    ROT_IMG = cv2.warpAffine(INP_IMG, rotation_matrix, (w, h))
    return(ROT_IMG, rotation_matrix)

def undistort_img(imCalRGB):
    ##############################################  Correcting lens distortion  ##############################################################
    img = imCalRGB.copy()
    h, w = img.shape[:2]
    camera_matrix = np.array([[w, 0, w/2],                                   ## Manually created without calibration data   
                                [0, h, h/2],                                   ## https://stackoverflow.com/questions/39432322/what-does-the-getoptimalnewcameramatrix-do-in-opencv
                                [0, 0, 1]], dtype=float)

    distortion_coeff = np.array([-0.1, -0.1, 0, 0, 0])      ## standard/typical values       ideally would be different for every camera
    #distortion_coeff = np.array([-0.5, -0.5, 0, 0, 0])      ## standard/typical values       ideally would be different for every camera
    #distortion_coeff = np.array([-0.5, -0.7, 0, 0, 0])      ## standard/typical values       ideally would be different for every camera

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeff, (w,h), 1, (w,h))
    undistorted_image = cv2.undistort(img, camera_matrix, distortion_coeff, None, new_camera_matrix)

    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y+h, x:x+w]           ## remove black edges created by correcting for distortion

    img = undistorted_image
    img = cv2.resize(img, (800,800))

    return(img)

def dart_transform(dart_orig_loc, transformations):
    dart_temp_loc = dart_orig_loc
    for matrix in transformations:

        p = dart_temp_loc   ## easier to write just 'p' 
    
        px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))   # how x coordinate transforms
        py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))   # how y coordinate transforms
        
        dart_temp_loc = [px, py]

    dart_final_loc = [px, py]
    return(dart_final_loc)

def find_centrepoint(lines, anglezone1, anglezone2, vertical_found = False, horizontal_found = False, counter = 0):
    global intersectpx, intersectpy, vertical_index, horizontal_index, x1, y1, x2, y2, x3, y3, x4, y4, img3
    points = []
    
    while counter < len(lines):                                                            ## Loop until we get through all lines in image
        if vertical_found == False or horizontal_found == False:                           ## Stop iterating through 'lines' when horiz and vert found
            rho, theta = lines[counter][0]    ## found line in polar coordinates
            print('count', counter, horizontal_found, vertical_found)
            
            if ((theta > anglezone1[0] and theta < anglezone1[1]) or (theta > anglezone1[2] and theta < anglezone1[3])) and vertical_found == False:
                #print('1st',  rho, theta)
                vertical_index = counter
                print('vertical found', vertical_index)
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 2000 * (-b))             ## plus 2000 just makes lines long enough to go off screen
                y1 = int(y0 + 2000 * (a))
                x2 = int(x0 - 2000 * (-b))
                y2 = int(y0 - 2000 * (a))

                cv2.line(img3,(x1,y1),(x2,y2),(0,255,0),1)   ## draws on each line found 
                vertical_found = True

            if ((theta > anglezone2[0] and theta < anglezone2[1]) or (theta > anglezone2[2] and theta < anglezone2[3])) and horizontal_found == False:
                #print('2nd', rho, theta)
                horizontal_index = counter
                print('horizontal found', horizontal_index)
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x3 = int(x0 + 3000 * (-b))
                y3 = int(y0 + 3000 * (a))
                x4 = int(x0 - 3000 * (-b))
                y4 = int(y0 - 3000 * (a))

                cv2.line(img3, (x3, y3), (x4, y4), (0, 0, 255), 1)
                horizontal_found = True
            #cv2.imshow('lines', img3)
            #cv2.waitKey(-1)
            counter += 1
        else:
            print( 'BREAK IN LOOP')
            break    

    points.append((x1,y1))    
    points.append((x2,y2))
    points.append((x3,y3))
    points.append((x4,y4))

    intersectpx, intersectpy = intersect(points[0], points[1], points[2], points[3])          
    return(intersectpx, intersectpy)

def intersect(p1, p2, p3, p4):                   ##https://gist.github.com/kylemcdonald/6132fc1c29fd3767691442ba4bc84018
    x1,y1 = p1
    x2,y2 = p2
    x3,y3 = p3
    x4,y4 = p4
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0: # parallel
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1: # out of range
        return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < 0 or ub > 1: # out of range
        return None
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return (x,y)

def calibrateboard(INP_IMG):

    global img, img2, img3, img4, warped

    img =  INP_IMG.copy()
    img2 = INP_IMG.copy()
    img3 = INP_IMG.copy()
    img4 = INP_IMG.copy()
    

    ##############################################  Making blurred gray image  ##############################################################
   
    #if np.shape(img) != (800, 800, 3):

        #img = cv2.resize(img, (800,800))
        #img2 = cv2.resize(img2, (800,800))
        #img3 = cv2.resize(img3, (800,800))
        #img4 = cv2.resize(img4, (800,800))


    img = cv2.GaussianBlur(img, (3,3), 0)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    

    ##### Cleaning up into black and white
    #ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU) ###  automatic otsu
    ret, thresh = cv2.threshold(gray,40,255,cv2.THRESH_BINARY_INV)    #(0 is typical good lower boundary)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilation = cv2.dilate(thresh, kernel, iterations=3)           ## expand white space
    erosion = cv2.erode(dilation, kernel, iterations=4)
   
    #######################################################################################################################################
    ####################################################### Finding outer ellipse #########################################################

    contours, hierarchy = cv2.findContours(erosion, 1, 1)           ## finds all the shapes
    contours_sorted = sorted(contours, key=cv2.contourArea)          ## sorts contours by area
   
    cv2.imshow('sdf', erosion)
    #cv2.waitKey(-1)
    cv2.destroyAllWindows()

    acceptable_circle = False
    for cnt in contours_sorted:
        if 1000 < cv2.contourArea(cnt):                                   ## ignore any really small contour
            
            cv2.drawContours(img2, cnt, -1, (0, 255, 0), 2)

            ellipse = cv2.fitEllipse(cnt)                                 ## fit an ellipse to current contour in loop
            cv2.ellipse(img2, ellipse, (255, 0, 0), 2)
           

            maj, minor = max(ellipse[1])/2, min(ellipse[1])/2             ## major and minor axis of eliipse
            ellipse_area = np.pi * maj * minor 
            area_sim = ellipse_area/cv2.contourArea(cnt)                  ## how close is shape's area to that of the fitted ellipse
            
           
            ellipse_perim = np.pi * (3 * (maj + minor) - np.sqrt((3*maj + minor)*(maj + 3*minor)))              ## estimating perimeter with Raj's formula
            contour_perim = cv2.arcLength(cnt, True)

            perim_sim = ellipse_perim/contour_perim                       ## how close is shape's perimeter to that of the fitted ellipse

    
            if (area_sim > 0.97 and area_sim < 1.03) and (perim_sim > 0.8 and perim_sim < 1.2):
                
                cv2.ellipse(img2, ellipse, (255, 0, 0), 2)
                x, y = ellipse[0]      ## centre of the ellipse
                axis1, axis2 = ellipse[1]      ## axis length
                ellipse_angle = ellipse[2]     ## ellipse rotation angle    https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga28b2267d35786f5f890ca167236cbc69
             
                center = (x, y)
                
                a = axis1/2           ## fixing axis length 
                b = axis2/2           ## fixing axis length 

                ##### plotting a box in around the ellipse ######
                rect = cv2.minAreaRect(cnt)    ## https://theailearner.com/tag/cv2-minarearect/
                box = cv2.boxPoints(rect)      ## boxPoints not BoxPoints
                box = np.intp(box)
                acceptable_circle = True

    print("found acceptable_circle", acceptable_circle)

    edges = cv2.Canny(dilation, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 120)   ## 1:input  2: rho  3: theta  4:requirement length     https://www.geeksforgeeks.org/line-detection-python-opencv-houghline-method/

    cv2.imshow('contours', img2)
    #cv2.waitKey(-1)

    ##################################################################################################################################################################
    ################################################  Segments for vertical and horizontal lines to be found in   ####################################################

    angle = ellipse_angle
    anglezone1 = [angle - 30, angle + 30]        ## creates segment of 20 degrees around angle of ellipse
    anglezone2 = [angle - 120, angle - 60]       ## creates segment of 20 degrees perp. to first segment


    anglezone1.append(anglezone1[0] + 180)        
    anglezone1.append(anglezone1[1] + 180)       
    anglezone2.append(anglezone2[0] + 180)
    anglezone2.append(anglezone2[1] + 180) 

    dummy = 0
    for angle in anglezone1:
        anglezone1[dummy] = angle * np.pi / 180 
        dummy += 1
    dummy = 0
    for angle in anglezone2:
        anglezone2[dummy] = angle * np.pi / 180 
        dummy += 1

    ##################################################################################################################################################################
    ################################################  Need to pick 2 near perpendicular lines to find intersection  ##################################################

    find_centrepoint(lines, anglezone1, anglezone2)       
    
    if (np.abs(intersectpx - x) > 50) and (np.abs(intersectpx - x) > np.abs(intersectpy - y)):                       ## stop if intersect isn't close to x-axis of ellipse              
        print('Problem with vertical line')
        find_centrepoint(lines, anglezone1, anglezone2, vertical_found=False, horizontal_found=True, counter = vertical_index + 1)   ## start function at next line after wrong line

    if (np.abs(intersectpy - y) > 50) and (np.abs(intersectpx - x) < np.abs(intersectpy - y)):                       ## stop if intersect isn't close to y-axis of ellipse
        print('Problem with horizontal line')  
        find_centrepoint(lines, anglezone1, anglezone2, vertical_found=True, horizontal_found=False, counter = horizontal_index + 1)   ## start function at next line after wrong line
    

    centre_coordinates = (int(intersectpx), int(intersectpy))
    cv2.circle(img3, centre_coordinates, 5, (255, 0, 0), -1)

    cv2.imshow('lines drawn in both segemnts', img3)  
    #cv2.waitKey(-1)        
    
    

    ################################################################################################################################################################
    ############################################ Creating a rotation matrix with angle of ellipse  #################################################################

  
    print('transforming image')

    if ellipse_angle > 90:
        ellipse_angle += 180

    #print(ellipse_angle)

    angle_rad = np.deg2rad(ellipse_angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])    ## https://en.wikipedia.org/wiki/Rotation_matrix

    axis1 = axis1/2 
    axis2 = axis2/2
    top = np.array([0, -axis2])   # Top point (along the minor axis)
    bottom = np.array([0, axis2]) # Bottom point (along the minor axis)
    left = np.array([-axis1, 0])  # Left point (along the major axis)
    right = np.array([axis1, 0])  # Right point (along the major axis)


    top_rotated = rotation_matrix.dot(top)
    bottom_rotated = rotation_matrix.dot(bottom)
    left_rotated = rotation_matrix.dot(left)
    right_rotated = rotation_matrix.dot(right)


    ##### Translate the points to the ellipse center
    top_vertex = (int(center[0] + top_rotated[0]), int(center[1] + top_rotated[1]))
    bottom_vertex = (int(center[0] + bottom_rotated[0]), int(center[1] + bottom_rotated[1]))                      
    left_vertex = (int(center[0] + left_rotated[0]), int(center[1] + left_rotated[1]))
    right_vertex = (int(center[0] + right_rotated[0]), int(center[1] + right_rotated[1]))
    cv2.circle(img2, centre_coordinates, 5, (255, 255, 255), -1)
    cv2.circle(img2, top_vertex, 5, (0, 0, 255), -1)     
    cv2.circle(img2, bottom_vertex, 5, (0, 255, 0), -1) 
    cv2.circle(img2, left_vertex, 5, (255, 0, 0), -1)    
    cv2.circle(img2, right_vertex, 5, (255, 255, 0), -1)   ### We now have 4 circles plotted at the top, bottom and sides of the ellipse ###

    source_pts = np.array([centre_coordinates, top_vertex, right_vertex, bottom_vertex, left_vertex], dtype = 'float32')    ## Creates anticlockwise selection of points


    ##### Stating the points of the circle we are transforming to
    CENTRE = [400,400]
    TOP = [400,60]
    RIGHT = [740,400]
    BOTTOM = [400,740]
    LEFT =  [60,400]
    cv2.circle(img2, CENTRE, 8, (255, 255, 255), -1)
    cv2.circle(img2, TOP, 8, (0, 0, 255), -1)     
    cv2.circle(img2, BOTTOM, 8, (0, 255, 0), -1) 
    cv2.circle(img2, LEFT, 8, (255, 0, 0), -1)    
    cv2.circle(img2, RIGHT, 8, (255, 255, 0), -1)

    ideal_box = np.array([CENTRE, TOP, RIGHT, BOTTOM, LEFT], dtype = 'float32')      ## circle points that we are warping to
    warp_matrix, nan = cv2.findHomography(source_pts, ideal_box)  ## finding warp matrix
    warped = cv2.warpPerspective(img4, warp_matrix, (800,800))

    cv2.imshow('warped', warped)
    #print('warp_matrix',warp_matrix)

    cv2.destroyAllWindows()

    return(warped, warp_matrix)

def get_calibration_data():

    cap  = cv2.VideoCapture(1)
    ret, prev_frame = cap.read()
    print('press \' j \' to select image')

    while (cap.isOpened()):
        ret, curr_frame = cap.read()
        cv2.imshow('cam', curr_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('j'):
            calibration_img = curr_frame
            break
    if not cap.isOpened():
        print("Cannot open camera")

    cv2.destroyAllWindows()

    #undistorted = undistort_img(calibration_img)
    undistorted = calibration_img
    cv2.imshow('undist', undistorted)

    calibration_list = []

    ################################### CALIBRATION ###################################

    warped, curr_matrix = calibrateboard(undistorted)
    calibration_list.append(curr_matrix)

    ################################ Calibrate and creat list of matrices ################################
    for i in range(0,5):
        
        cv2.imshow('adsf', warped)
        #cv2.waitKey(-1)
        new_board = warped 

        warped, curr_matrix = calibrateboard(new_board)
        calibration_list.append(curr_matrix)
                            

    test = warped

    #player_input = 10
    while clicked_point == None:
        #cv2.line(test, (400, 800), (400, 0), (255,0,0), 2)
        cv2.imshow('test', test)
        cv2.setMouseCallback('test', mouse_callback)
        cv2.waitKey(-1)
        cv2.destroyAllWindows()
        
        x = clicked_point[0] - 400  ## distance in the x direction
        y = 400 - clicked_point[1]  ## distance in the y direction      (reversed because y axis increases downwards)

        rot_angle = np.degrees(np.arctan2(x, y)) 
        #print(rot_angle)

        test, rot_mat = rotate_img(test, rot_angle)              ## rotate by amount inputted anticlockwise
        rot_mat_homog = np.vstack([rot_mat, (0,0,1)])
        calibration_list.append(rot_mat_homog)                    ## add rotation matrix to list

        if key == ord('q'):
            break

    np.save('calibration_data.npy', calibration_list)

###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################

def drawBoard(INPUT_IMAGE):
    #raw_loc_mat = np.zeros((800, 800, 3))
    raw_loc_mat = INPUT_IMAGE.copy()

    # draw board
    cv2.circle(raw_loc_mat, (400, 400), 170 * 2, (255, 255, 255), 1)  # outside double
    cv2.circle(raw_loc_mat, (400, 400), 160 * 2, (255, 255, 255), 1)  # inside double
    cv2.circle(raw_loc_mat, (400, 400), 107 * 2, (255, 255, 255), 1)  # outside treble
    cv2.circle(raw_loc_mat, (400, 400), 97 * 2, (255, 255, 255), 1)  # inside treble
    cv2.circle(raw_loc_mat, (400, 400), 16 * 2, (255, 255, 255), 1)  # 25
    cv2.circle(raw_loc_mat, (400, 400), 7 * 2, (255, 255, 255), 1)  # Bulls eye

    # 20 sectors...
    sectorangle = 2 * math.pi / 20
    i = 0
    while (i < 20):
        cv2.line(raw_loc_mat, (400, 400), (
            int(400 + 170 * 2 * math.cos((0.5 + i) * sectorangle)),
            int(400 + 170 * 2 * math.sin((0.5 + i) * sectorangle))), (255, 255, 255), 1)
        i += 1

    return raw_loc_mat

def dart_transform(dart_orig_loc, transformations):
    dart_temp_loc = dart_orig_loc
    for matrix in transformations:

        p = dart_temp_loc   ## easier to write just 'p' 
    
        px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))   # how x coordinate transforms
        py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))   # how y coordinate transforms
        
        dart_temp_loc = [px, py]

    dart_final_loc = [px, py]
    return(dart_final_loc)

def get_dart_position(dart_image):
    dilate = dart_image.copy()
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    first = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel, iterations=1)     #####    removes little white dots that are noise
    final = cv2.dilate(first, kernel, iterations=1) 

    contours, hierarchy = cv2.findContours(final, 1, 1) 
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    centers_list = [] 

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

    #print(centers_list)
    #cv2.imshow('centers', diff)
    #cv2.waitKey(-1)

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

    return(rightmost_point)

def get_dart_score(dart_image, result_queue):
    #global curr_frame

    calibration_list = np.load('calibration_data.npy', allow_pickle=True)
    dilate = dart_image.copy()

    dart_tip = get_dart_position(dilate)

    dart_loc = np.intp(dart_transform((dart_tip), calibration_list))          ## uses dart_transform function to get actual position

    blank = np.zeros((800, 800, 3))
    board= drawBoard(blank)
    
    #cv2.circle(curr_frame, dart_tip, 5, (255,0,0), -1)
    cv2.circle(board, dart_loc, 5, (255,0,0), -1)
        
    score = LOC2SCORE(dart_loc)                   ## turns the actual dart position into a score
    print('score', score)

    cv2.imshow('dart', dilate)
    #cv2.imshow('board', curr_frame)
    
    #cv2.waitKey(-1)
    #cv2.destroyAllWindows()
    result_queue.put(score)
    return(score)

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

def find_noise():
    feed  = cv2.VideoCapture(cam_feed_num)

    #cv2.namedWindow("Main", cv2.WINDOW_NORMAL)  ## allows window to be made bigger

    ret, prev_frame = feed.read()              ## Get an intial frame
    
    frame_count = 0
    half_second_loops = 0
    noise_list = []
    while(feed.isOpened()):
        # Capture each frame
        ret, curr_frame = feed.read()              ##  Get a new frame
        curr_frame = cv2.rotate(curr_frame, cv2.ROTATE_180)
        frame_count += 1
        cv2.imshow('Main', curr_frame)
       
        if ret == True:
            if frame_count > 29:                                             ## wait "" frames before executing this loop
                diff = cv2.absdiff(prev_frame, curr_frame)
                diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

                _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
                thresh = cv2.GaussianBlur(thresh, (5, 5), 0)    ##  heavy blur to reduce noise??
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)     #####    removes little white dots that are noise
                dilate = cv2.dilate(open, kernel, iterations=3)   #### expands what is lef twhich should onyl be a dart as its large enough to not be removed from prev step


                cv2.imshow('diff', diff)
                cv2.imshow('dilate', dilate)

                half_second_loops += 1
                frame_count = 0             ## Reset new checking period
                prev_frame = curr_frame 
            else:
                continue
        
            change = np.sum(dilate) / 255

            if half_second_loops > 8 < 14:
                print(change)
                noise_list.append(change)

            if half_second_loops > 10:
                break


            key = cv2.waitKey(1)
            if key == ord('q'):
                break
 
    total_noise = 0
    for change in noise_list:
        total_noise += change

    average_noise = total_noise/len(noise_list)
    threshold1 = 1.3 * average_noise
    threshold2 = 500 + average_noise

    threshold = max(threshold1, threshold2)

    return(threshold)
            

darts_detected = 0
player_next_ready = False
curr_player = 1
P1_total = 501
P2_total = 501


#threshold = find_noise()
#print(f'threshold set at {threshold}')

cv2.destroyAllWindows()

def live_detection():
    global darts_detected, player_next_ready, curr_player, P1_total_score, P2_total_score, P1_total, P2_total, threshold, curr_player
    #threshold = find_noise()             ### optional find backgfroudn noisea in image
    threshold = 1700
    #print(f'threshold set at {threshold}')
    #while not threshold:
    #   print('waiting for threshold')

    
    feed = cv2.VideoCapture(cam_feed_num)

    #cv2.namedWindow("Main", cv2.WINDOW_NORMAL)  ## allows window to be made bigger
    ret, prev_frame = feed.read()              ## Get an intial frame

    frame_count = 0
    checked = 0             ## current number of comparisons run, used to stop comparison before focussing/setup occurs 
    curr_player = 1 
    while(feed.isOpened()):
         
        ret, curr_frame = feed.read()      ##  Get a new frame ideally every 1/fps 
        #curr_frame = cv2.rotate(curr_frame, cv2.ROTATE_180)
        frame_count += 1
        cv2.imshow('Main', curr_frame)
        if ret == True:
            check_frequency_frames = 30
            if frame_count > check_frequency_frames:          ## wait "" frames before executing this loop
                #cv2.imwrite(f'checked{checked}.jpg', curr_frame)             
                #_, safety_frame = feed.read() 
                diff = cv2.absdiff(prev_frame, curr_frame)
                diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

                _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
                ##thresh = cv2.GaussianBlur(thresh, (5, 5), 0)    ##  heavy blur to reduce noise??
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)     #####    removes little white dots that are noise
                dilate = cv2.dilate(open, kernel, iterations=3) 

        
                checked += 1                ## increments by 1 for the comparison loop
                frame_count = 0             ## Reset new checking period
                still_dartboard_img = prev_frame    ## save the still dartboard to be later compared with dart in the board
                prev_frame = curr_frame     ## make the current frame the next loop's previous frame, i.e still awaiting change/dart d4etection
            
            else:
                continue                ## skips the rest of the loop
            

    ##############################################################################################
    #####################   THE BELOW CODE IS ONLY REACHED EVERY X FRAMES #######################
        
            change = np.sum(dilate) / 255      ### Calculate how different the 2 imgs were after processing

            if change > threshold and checked > 3:   ## ignores initial focussing difference using "checked" variable. DART IS DETECTED
                darts_detected += 1
                
                
                cv2.imwrite(f'0000  dilate  {darts_detected}.jpg', dilate)
                cv2.imwrite(f'0000  diff  {darts_detected}.jpg', diff)
          
           
            
              
                ##### Sometimes it will detect dart mid flight which is fine for triggering detecting a dart but not for location of dart ######
                ##### If we wait a few frames after detecting a change then we should fix this error #####
            
                buffer = 0
                while buffer < 5:
                    _ , buffer_frame = feed.read()      
                    buffer += 1                                        
                
                ### now we want to compare the buffer frame to the still dartboard image we detected a change from which is "still_dartboard_img"

                ################### Same processing for change but with still img and dart img #####################
                diff_2 = cv2.absdiff(still_dartboard_img, buffer_frame)
                diff_2 = cv2.cvtColor(diff_2, cv2.COLOR_RGB2GRAY)
                _, thresh_2 = cv2.threshold(diff_2, 15, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                open_2 = cv2.morphologyEx(thresh_2, cv2.MORPH_OPEN, kernel, iterations=2)     #####    removes little white dots that are noise
                dilate_2 = cv2.dilate(open_2, kernel, iterations=3)
                ####################################################################

            
                #########  Start thread for finding dart location --> transform it --> get score  #########

                dart_thread = threading.Thread(target=get_dart_score, args=(dilate_2.copy(), result_queue ))
                dart_thread.start()

                #dart_found = threading.Thread(target=update_label(curr_player, darts_detected), args=(dilate_2.copy(), result_queue ))
                #dart_found.start()

                key = f'P{curr_player}D{darts_detected}'
                widgets[key].config(highlightbackground="green")

                #dart_thread.join()
                #dart_score = result_queue.get()
                #player_score.append(dart_score)
                #cv2.imshow('diff', diff)
                #cv2.imshow('dilate', dilate)
                #cv2.imwrite(f'dart{darts_detected}.jpg', curr_frame)  
                #cv2.imwrite(f'dart{darts_detected}_diff.jpg', diff)  
                #cv2.imwrite(f'dart{darts_detected}_dilate.jpg',dilate) 

            prev_frame = curr_frame       ## make the current frame the next loop's previous frame

            if darts_detected >= 3:
                print('Remove darts and then press any key for next turn')
                
                while not player_next_ready:                   ### Wait until players removes darts before next player is ready
                    pass
                
                if player_next_ready:
                    player_next_ready = False                  ## Set back to default
                    
                    ret, curr_frame = feed.read()       ## Darts will have been removed which will resest dartbaord to blank
                    prev_frame = curr_frame
                    frame_count += 1                    ## Increment frame count for differece check
                    checked = 3                         ## Set to 3 for nexct loop as no more focussing should occur
                    darts_detected = 0
                    ################### switch the player, reduce player total by 3 score boxes ###########################
                    if curr_player == 1:                
                        P1_total_score = int(widgets['P1D1'].cget('text')) + int(widgets['P1D2'].cget('text')) + int(widgets['P1D3'].cget('text'))
                        P1_total -= P1_total_score
                        P1_edit_total(widgets["score1"])
                        curr_player = 2
                    else:
                        P2_total_score = int(widgets['P2D1'].cget('text')) + int(widgets['P2D2'].cget('text')) + int(widgets['P2D3'].cget('text'))
                        P2_total -= P2_total_score
                        P2_edit_total(widgets["score2"])
                        curr_player = 1
                    print(f'Ready for player{curr_player}')


            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('p'):
                cv2.waitKey(-1)

            if key ==  ord('s'):
                cv2.imwrite()



#calibration_list = np.load('calibration_data.npy', allow_pickle=True)
#score_queue = queue.Queue()

###  If board appears like a corrrectly rotated board then continue  ###

##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################

window = tk.Tk()
#window.geometry('800x600')

def button_command1():
    name = input.get()
    print('test', name)
    return None

def button_command2():
    name = input.get()
    print('test', name)
    return None

def update_p1():
    entered_text = input.get() 
    player1.config(text=entered_text)

def update_p2():
    entered_text = input.get()  
    player2.config(text=entered_text)

def calibrate():
    print('calibrating')
    get_calibration_data()

def start_game():
    #global 
    print('starting game')

    tracking_thread = threading.Thread(target=live_detection)
    tracking_thread.start()

def next_player():
    global player_next_ready
    #key = f'P{curr_player}D{darts_detected}'
    player_next_ready = True
    highlight_player(curr_player)
    
    widgets[f'P{curr_player}D1'].config(highlightbackground="orange")                        ## resets all the players dart scores back to original from green
    widgets[f'P{curr_player}D2'].config(highlightbackground="orange")
    widgets[f'P{curr_player}D3'].config(highlightbackground="orange")

def edit_score(target_label):
    global darts_detected
    if darts_detected < 3:
        darts_detected += 1
    key = f'P{curr_player}D{darts_detected}'
    entered_text = edit_score_entry.get() 
    widgets[key].config(highlightbackground="green") 
    target_label.config(text=entered_text)         ## this way whatever is inputted into the function is the label changed
    edit_score_entry.delete(0, tk.END)             ## clear the entry box
    if darts_detected < 3:
        darts_detected += 1

def P1_edit_total(target_label):
    global P1_total
    print(P1_total)
    target_label.config(text=P1_total)         ### this way whatever is inputted into the function is the label changed

def P2_edit_total(target_label):
    global P2_total
    print(P2_total)
    target_label.config(text=P2_total)         ### this way whatever is inputted into the function is the label changed

def update_score():
    global result_queue, darts_detected, curr_player
    try:
        while not result_queue.empty():
            #print('player ',curr_player, 'darts detec ',darts_detected)

            score = result_queue.get_nowait()
            key1 = f'score{curr_player}'
            #widgets[key1].config(text=f"Score: {score}")
            dart_being_detected = darts_detected #+ 1
            #print(dart_being_detected)
            key2 = f'P{curr_player}D{dart_being_detected}'
            widgets[key2].config(text=f"{score}")
            
    except queue.Empty:
        pass
    window.after(100, update_score)   ## check again after 100ms

greeting = tk.Label(
    text='Welcome to Auto-Darts',
    bg='black',
    fg='white',
    font=("Arial", 30),
    width=40,
    height=3
)

button1 = tk.Button(
    text="Submit name",
    command=update_p1,
    width=12,
    height=1,
    font=("Arial", 16),
    bg="orange",
    fg="black",
)

button2 = tk.Button(
    text="Submit name",
    command=update_p2,
    width=12,
    height=1,
    font=("Arial", 16),
    bg="orange",
    fg="black",
)

button3 = tk.Button(
    text="Calibrate",
    command=calibrate,
    width=25,
    height=2,
    font=("Arial", 16),
    bg="black",
    fg="white",
)

button4 = tk.Button(
    text="Start Game!!",
    command=start_game,
    width=25,
    height=2,
    font=("Arial", 16),
    bg="black",
    fg="white",
)

input = tk.Entry(
    width=30,
    bg="white",
    fg="red",
)

edit_score_entry = tk.Entry(
    width=10,
    bg="white",
    fg="red",
    bd=4,
    font=("Helvetica", 30)
)

player1 = tk.Label(
    text='Player 1',
    bg='yellow',
    fg='white',
    font=("Arial", 26),
    width=20,
    height=2
)

player2 = tk.Label(
    text='Player 2',
    bg='lightgrey',
    fg='white',
    font=("Arial", 26),
    width=20,
    height=2
)

next_player_button = tk.Button(
    text='Next Player',
    command=next_player,
    bg='black',
    fg='white',
    font=("Arial", 26),
    width=20,
    height=2
)

widgets = {}
def setup_widgets(window):
    global widgets

    widgets["P1D1"] = tk.Label(text='1',bg='white',fg='blue',highlightbackground="orange",highlightthickness=10,font=("Arial", 40),width=4,height=2)
    widgets["P1D2"] = tk.Label(text='2',bg='white',fg='blue',highlightbackground="orange",highlightthickness=10,font=("Arial", 40),width=4,height=2)
    widgets["P1D3"] = tk.Label(text='3',bg='white',fg='blue',highlightbackground="orange",highlightthickness=10,font=("Arial", 40),width=4,height=2)
    widgets["P2D1"] = tk.Label(text='1',bg='white',fg='red',highlightbackground="orange",highlightthickness=10,font=("Arial", 40),width=4,height=2)
    widgets["P2D2"] = tk.Label(text='2',bg='white',fg='red',highlightbackground="orange",highlightthickness=10,font=("Arial", 40),width=4,height=2)
    widgets["P2D3"] = tk.Label(text='3',bg='white',fg='red',highlightbackground="orange",highlightthickness=10,font=("Arial", 40),width=4,height=2)
    widgets["score1"] = tk.Label(text=f'{P1_total}',bg='blue',fg='white',font=("Arial", 120),width=8,height=2)
    widgets["score2"] = tk.Label(text=f'{P2_total}',bg='red',fg='white',font=("Arial", 120),width=8,height=2)
    widgets["Edit P1D1"] = tk.Button(text='',command=lambda: edit_score(widgets["P1D1"]),bg='orange',fg='blue',highlightbackground="orange",highlightthickness=5,font=("Arial", 26),width=6,height=1)
    widgets["Edit P1D2"] = tk.Button(text='',command=lambda: edit_score(widgets["P1D2"]),bg='orange',fg='blue',highlightbackground="orange",highlightthickness=5,font=("Arial", 26),width=6,height=1)
    widgets["Edit P1D3"] = tk.Button(text='',command=lambda: edit_score(widgets["P1D3"]),bg='orange',fg='blue',highlightbackground="orange",highlightthickness=5,font=("Arial", 26),width=6,height=1)
    widgets["Edit P2D1"] = tk.Button(text='',command=lambda: edit_score(widgets["P2D1"]),bg='orange',fg='red',highlightbackground="orange",highlightthickness=5,font=("Arial", 26),width=6,height=1)
    widgets["Edit P2D2"] = tk.Button(text='',command=lambda: edit_score(widgets["P2D2"]),bg='orange',fg='red',highlightbackground="orange",highlightthickness=5,font=("Arial", 26),width=6,height=1)
    widgets["Edit P2D3"] = tk.Button(text='',command=lambda: edit_score(widgets["P2D3"]),bg='orange',fg='red',highlightbackground="orange",highlightthickness=5,font=("Arial", 26),width=6,height=1)

setup_widgets(window)

## making update function that can be used with different labels 
def update_label(player, dart):
    key = f'P{player}D{dart}'
    entered_text = edit_score.get()
    if key in widgets:
        widgets[key].config(text=entered_text)
        widgets[key].config(highlightbackground="green")


def highlight_player(curr_player):
    if curr_player == 1:
        player2.config(bg="yellow")  # Highlight Player 1
        player1.config(bg="lightgray")  # Reset Player 2
    elif curr_player == 2:
        player2.config(bg="lightgray")  # Reset Player 1
        player1.config(bg="yellow")  # Highlight Player 2

############################################################################

greeting.grid(row=0, column=0, columnspan=7, sticky='ew')
input.grid(row=1, column=3, columnspan=1)
button1.grid(row=1, column=1, columnspan=1)
button2.grid(row=1, column=5, columnspan=1)
button3.grid(row=2, column=3, sticky='ew' 'ns')
button4.grid(row=3, column=3, sticky='ew' 'ns')
player1.grid(row=2, column=0, columnspan=3, sticky='ew')
player2.grid(row=2, column=4, columnspan=3, sticky='ew')
next_player_button.grid(row=4, column=3, columnspan=1, sticky='ew')
widgets['score1'].grid(row=3, column=0, columnspan=3, sticky='ew')
widgets['score2'].grid(row=3, column=4, columnspan=3, sticky='ew')
widgets['P1D1'].grid(row=4, column=0, sticky='ew')
widgets['P1D2'].grid(row=4, column=1, sticky='ew')
widgets['P1D3'].grid(row=4, column=2, sticky='ew')
widgets['P2D1'].grid(row=4, column=4, sticky='ew')
widgets['P2D2'].grid(row=4, column=5, sticky='ew')
widgets['P2D3'].grid(row=4, column=6, sticky='ew')
edit_score_entry.grid(row=5, column=3, sticky='ew')
widgets['Edit P1D1'].grid(row=5, column=0, sticky='ew')
widgets['Edit P1D2'].grid(row=5, column=1, sticky='ew')
widgets['Edit P1D3'].grid(row=5, column=2, sticky='ew')
widgets['Edit P2D1'].grid(row=5, column=4, sticky='ew')
widgets['Edit P2D2'].grid(row=5, column=5, sticky='ew')
widgets['Edit P2D3'].grid(row=5, column=6, sticky='ew')


result_queue = queue.Queue()
window.after(100, update_score)
#window.after(100, update_P{}D{darts_de})

window.mainloop()


##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################



#cap.release()
cv2.destroyAllWindows()
