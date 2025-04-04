#import cv2.version
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




clicked_point = None
def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"Clicked point: {clicked_point}")

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

def rotate_img(INP_IMG, anti_clock_rot_deg):
    
    h, w = np.shape(INP_IMG)[:2]
    center = (w/2, h/2)
    rotation_matrix = cv2.getRotationMatrix2D(center, anti_clock_rot_deg, 1)
    ROT_IMG = cv2.warpAffine(INP_IMG, rotation_matrix, (w, h))
    return(ROT_IMG, rotation_matrix)

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

def calibrateboard(INP_IMG):

    global img, img2, img3, img4, warped 

    img =  INP_IMG.copy()
    img2 = INP_IMG.copy()
    img3 = INP_IMG.copy()
    img4 = INP_IMG.copy()
    img4 = INP_IMG.copy()
    
    ##############################################  Making blurred gray image  ##############################################################
    
    img = cv2.GaussianBlur(img, (3,3), 0)                     ## Blurs the image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)               ## Converts to grayscale so its ready for thresholding
    

    ###########################################  Cleaning up into black and white  ###########################################################

    #ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU) ##  automatic otsu
    ret, thresh = cv2.threshold(gray,40,255,cv2.THRESH_BINARY_INV)    ##  Converts to black and white binary img,  0 is typical good lower boundary

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))     ## Structure of the kernel thats used in dilation and erosion
    dilation = cv2.dilate(thresh, kernel, iterations=3)           ## expand white space
    erosion = cv2.erode(dilation, kernel, iterations=4)           ## 
   
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
    global calibration_list , calibration_img, test

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
    for i in range(0,10):
        
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

    #np.save('test_calib_data.npy', calibration_list)



get_calibration_data()


#transformed_dartboard = cv2.warpAffine(calibration_img, start, (800,800))



cv2.imshow('original', calibration_img)
cv2.imshow('final', test)
cv2.imshow('final', drawBoard(test))


cv2.imwrite('calibration pic.jpg', calibration_img)
cv2.imwrite('transformed.jpg', test)
cv2.imwrite('transformed over ideal board.jpg', drawBoard(test))



cv2.waitKey(-1)

cv2.destroyAllWindows()