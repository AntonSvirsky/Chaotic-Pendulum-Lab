# -*- coding: utf-8 -*-
"""
Script Name: dp_traj_extract_main.py

Description:
    This script extracts the (x,y) coordinates of two masses (m1,m2) from a 
    video of a double pendulum. The script produces a .csv file with time and 
    (x,y) position data for both masses. An interpolation option is implemented
    to fix discontinuites in the data. Additionaly, an analysis movie is 
    generated to manualy inspect the accuracy of the track.

Usage:
    Run the script with the input and output parameters. 

Dependencies:
    - numpy
    - pandas
    - matplotlib
    - opencv-python

Author: Anton S.
Date: 2025-06-12
Version: 1.1
"""

#################################################################
#                                                               #
#                        PARAMETERS                             #
#                                                               #
#################################################################

#Input parameters
video       ='100deg5'      #name of the movie to analyse without extension
video_name  = video+'.MOV';
vid_dir     = 'C:\\CP_Analysis\\movies'; #directory with the movies
fps         = 120                        #the actual FPS of the video

#Output parameters
analysis_dir   = 'C:\\CP_Analysis\\movies\\analysis'; #output directory
time_stamp     = True;  # Create a new analysis folder for each run
interpolate    = True;  # Liniarly interpolate missing points
output_movie   = True;  # Write a video with the marked object for validation
seperate_files = False; # Creates a seperate data file for each mass
flip_y         = False; #flip the y axis
#Misc. parameters
color_m1 = (255,0,0);    # color of mass M1
color_m2 = (0,255,0);    # color of mass M2
out_video_scale_down = 2 # scaling down factor of output video (smaller file)


#################################################################
#                                                               #
#                    CODE STARTS HERE                           #
#                                                               #
#################################################################

#VERSION 2.0

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import pandas as pd

import dp_extraction_utils as dp #costum utility functions

#create ouput dir
folder_name = os.path.splitext(video_name)[0] #take video name as folder name

if time_stamp: #add time stamp
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{folder_name}_{current_time}"

out_dir = os.path.join(analysis_dir,folder_name) #output directory
os.makedirs(out_dir, exist_ok=True) #make directory if it dose not exist

#create object color vector
object_colors = [color_m1,color_m2] #colors for markers


# Callback function for mouse events
def click_event(event, x, y, flags, params):
    global origin
    if event == cv2.EVENT_LBUTTONDOWN:
        # Store the coordinates
        origin = (x, y)
        print(f"Origin at: {origin}")
        
        # Draw a circle at the point clicked
        frame_with_cross  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_with_cross  = cv2.cvtColor(frame_with_cross, cv2.COLOR_GRAY2BGR)
        # Draw the crosshair at the clicked point
        dp.draw_crosshair(frame_with_cross, origin)
        cv2.imshow("Select origin (Enter to confirm)", frame_with_cross)


# Initialize the global variable to store the coordinates of the origin
origin = (0,0)


# Open the input video
video_path = os.path.join(vid_dir, video_name)
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    sys.exit(1)
else:
    print("Video loaded sucsessfult")
    
#create output video
if output_movie:
    output_video_path =os.path.join(out_dir,'tracking_video.mp4')
    out_video,output_resolution = dp.create_output_movie(cap, output_video_path,scale_down = out_video_scale_down)    
    
    
#open first frame 
ret, frame = cap.read()

# Display the image
cv2.imshow("Select origin (Enter to confirm)", frame)
# Set the mouse callback function
cv2.setMouseCallback("Select origin (Enter to confirm)", click_event)

# Wait for Enter to be pressed
k = cv2.waitKey(100)
while k != 13:
    k = cv2.waitKey(100)
    
    if cv2.getWindowProperty('Select origin (Enter to confirm)',cv2.WND_PROP_VISIBLE) < 1:        
        # Display the image in case it was closed
        cv2.imshow("Select origin (Enter to confirm)", frame)

# Close all OpenCV windows
cv2.destroyAllWindows()

#select the first mass
roi_m1 = cv2.selectROI("Select M1 (Press Eneter to confirm)", frame, fromCenter=False, showCrosshair=True);
roi_m2 = cv2.selectROI("Select M2 (Press Eneter to confirm)", frame, fromCenter=False, showCrosshair=True);

cv2.destroyAllWindows()

# Extract the mean HSV values from the selected ROI
hsv_m1 =  dp.get_avg_hsv_from_roi(frame,roi_m1)
hsv_m2 =  dp.get_avg_hsv_from_roi(frame,roi_m2)

# Show the mean hsv values
print("Average HSV valuess:")
print(f"  M1: {hsv_m1}")
print(f"  M2: {hsv_m2}")

#construct filters for points of interest
filt_m1 =  dp.construct_filter(hsv_m1,h_margin = 10, s_margin = 75, v_margin = 75)
filt_m2 =  dp.construct_filter(hsv_m2,h_margin = 10, s_margin = 75, v_margin = 75)

#construct detection image
calib_frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
dp.draw_crosshair(calib_frame, origin)
for i,filt in enumerate([filt_m1, filt_m2]):
    _,_,_,bounding_rect = dp.detect_color(frame,filt)
    dp.add_marker(calib_frame, bounding_rect, object_colors[i])

# construct, show and save image
fig_cal = plt.figure()
plt.imshow(cv2.cvtColor(calib_frame, cv2.COLOR_BGR2RGB))
plt.title(video_name)
plt.show()
fig_cal.savefig(os.path.join(out_dir,'detection_image.png')) 


#empty positions and found indicators
frame_count = 0
positions_m1  = [] 
positions_m2  = [] 
found_indicator_m1 = [] 
found_indicator_m2 = [] 


while ret:
    cx_1,cy_1,obj_found_1,bbox_1 = dp.detect_color(frame,filt_m1,min_size =100)
    cx_2,cy_2,obj_found_2,bbox_2 = dp.detect_color(frame,filt_m2,min_size =100)

    time = frame_count / fps
    
    positions_m1.append([time, cx_1 - origin[0], cy_1 - origin[1]])
    positions_m2.append([time, cx_2 - origin[0], cy_2 - origin[1]])
    
    found_indicator_m1.append(obj_found_1)
    found_indicator_m2.append(obj_found_2)
    
    #write output video frame
    if output_movie:
        out_frame = dp.create_output_frame(frame, bbox_1, obj_found_1, 
                                           bbox_2, obj_found_2, object_colors,
                                           output_resolution, origin)
        out_video.write(out_frame)

        
    if frame_count % 500 == 0 :
        print(f'Frame {frame_count}: Found objects '+
              f'm1 : {np.sum(found_indicator_m1)}/{frame_count+1} '+
              f'm2 : {np.sum(found_indicator_m2)}/{frame_count+1} ')
    #read new frame
    ret, frame = cap.read()
    frame_count += 1

#release resorces
cap.release() # close input video 

if output_movie: #close output video
    out_video.release()


#convert to vectors
pts_m1 = np.array(positions_m1)
pts_m2 = np.array(positions_m2)

#extract array (Note physical y coorinates are fliped w.r.t image coordinate)
if flip_y:
    t_1,cx_1,cy_1 =pts_m1[:,0], pts_m1[:,1], -pts_m1[:,2]
    t_2,cx_2,cy_2 =pts_m2[:,0], pts_m2[:,1], -pts_m2[:,2]
else:
    t_1,cx_1,cy_1 =pts_m1[:,0], pts_m1[:,1], pts_m1[:,2]
    t_2,cx_2,cy_2 =pts_m2[:,0], pts_m2[:,1], pts_m2[:,2]

#interpolate missing points
if interpolate:
    cx_1 = np.interp(t_1, t_1[found_indicator_m1], cx_1[found_indicator_m1])
    cy_1 = np.interp(t_1, t_1[found_indicator_m1], cy_1[found_indicator_m1])
    cx_2 = np.interp(t_2, t_2[found_indicator_m2], cx_2[found_indicator_m2])
    cy_2 = np.interp(t_2, t_2[found_indicator_m2], cy_2[found_indicator_m2])
    
#save t,x,y data fo
if seperate_files:
    #seperate files
    #create data frame
    dat_m1 = {'Time': t_1, 'X': cx_1, 'Y': cy_1}
    dat_m2 = {'Time': t_2, 'X': cx_2, 'Y': cy_2}
    
    df_m1 = pd.DataFrame(dat_m1)
    df_m2 = pd.DataFrame(dat_m2)
    
    # Write the DataFrame to a CSV files
    df_m1.to_csv(os.path.join(out_dir,'m1_data.csv'), index=False)
    df_m2.to_csv(os.path.join(out_dir,'m2_data.csv'), index=False)
else: 
    #single file
    #create data frame
    dat = {'Time': t_1, 'X1': cx_1, 'Y1': cy_1, 'X2': cx_2, 'Y2': cy_2}
    df = pd.DataFrame(dat)
    # Write the DataFrame to a CSV files
    df.to_csv(os.path.join(out_'data.csv'), index=False)

#finish program
sys.exit(0)