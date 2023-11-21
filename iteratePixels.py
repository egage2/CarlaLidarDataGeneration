import cv2
import numpy as np
import os
import time
import glob


def get_earliest_frame(datapath):
    earliest_frame = 999999
    num_files = 0
    for filename in os.listdir(datapath):
        file = os.path.join(datapath,filename)
        if os.path.isfile(file):
            
            if file.endswith('.png'):
                num_files +=1
                frame = file.strip(datapath)
                
                frame = frame.strip('.png')
                
                if int(frame) < earliest_frame:
                    earliest_frame = int(frame)
    return(earliest_frame,num_files)


def display_camera_data(filepath, earliest_frame, num_files):
    img_arr = []
    frame = earliest_frame
    for i in range(num_files):
        data_path = filepath + '/' + str(frame) + '.png'
        frame += 1
        print("Opening File: " + data_path)
        img = cv2.imread(data_path)
        height = img.shape[0]
        width = img.shape[1]
        size = (width,height)
        carList = []

        for i in range(height):
            for j in range(width):
                #If semantic tag == 4 or 10, for pedestrian or vehicle
                pixel = []
                    
                #OpenCV stores images in BGR not RGB
                if img[i,j][2] == 4 or img[i,j][2] == 10:
                    #return pixel rgb value
                    
                    #Empty carlist
                    if len(carList) == 0:
                        pixel = img[i,j]
                        carList.append(pixel)
                    else:
                        isUnique = True
                        for k in carList:
                            if k[0] == img[i,j,0] and k[1] == img[i,j,1] and k[2] == img[i,j,2]:
                                isUnique = False
                        if isUnique:
                            pixel = img[i,j]
                            carList.append(pixel)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_list = []
        contours_list = []
        #For every can in car List, make HSV mask, mask image, draw contours,
        #add contours onto base image
        for i in range(len(carList)):
            
            np_arr = np.uint8([[[carList[i][0], carList[i][1], carList[i][2]]]])
            hsv_mask = cv2.cvtColor(np_arr, cv2.COLOR_BGR2HSV)
            
            lower = np.array([hsv_mask[0,0,0]-1,hsv_mask[0,0,1]-1,hsv_mask[0,0,2]-1])
            upper = np.array([hsv_mask[0,0,0]+1,hsv_mask[0,0,1]+1,hsv_mask[0,0,2]+1])
            mask = cv2.inRange(hsv,lower,upper)
            
            mask_list.append(mask)
            contour,heirarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            contours_list.append(contour)
            #Draw red contours for peds(4), green contours for cars(10)
            if carList[i][2] == 4:
                #BGR
                cv2.drawContours(img, contour, -1, (0,0,255),1)
            elif carList[i][2] == 10:
                cv2.drawContours(img, contour, -1, (0,255,0),1)
            
        img_arr.append(img)
        #cv2.imshow("Frame", img)
        #cv2.waitKey(0)
    return(img_arr,size)



datapath = "carlaScenarioRunner/test_data17/camera_data"
frame, numfiles = get_earliest_frame(datapath)
images,size = display_camera_data(datapath,frame,numfiles)

out = cv2.VideoWriter('test.avi',cv2.VideoWriter_fourcc(*'MP4V'),15,size)
for i in range(len(images)):
    out.write(images[i])
out.release()