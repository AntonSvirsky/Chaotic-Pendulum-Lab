"""
Script Name: dp_extraction_utils.py

Description:
    Auxillary functions used by the dp_traj_extract_main.py. 

Usage:
    Functions are called by dp_traj_extract_main.py
    

Dependencies:
    - numpy
    - opencv-python

Author: Anton S.
Date: 2025-06-12
Version: 1.0
"""

import cv2
import numpy as np


def create_output_movie(cap,output_video_path, scale_down = 2):
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    output_resolution = (frame_width//scale_down, frame_height//scale_down) 
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, output_resolution)
    return out_video,output_resolution


def add_marker(frame, bounding_box, color):
    x, y, w, h = bounding_box
    cx, cy = x + w//2, y+h//2
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)  # Bounding box
    cv2.circle(frame, (cx, cy), 5, color, -1)  # Centroid
    return

def create_output_frame(frame, box_1, found_1, box_2, found_2,colors,output_resolution,origin): 
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame_colored = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    add_marker(gray_frame_colored, box_1, (0,0,255))
    draw_crosshair(gray_frame_colored, origin)
    if found_1: 
        add_marker(gray_frame_colored, box_1, colors[0])
    
    if found_2: 
        add_marker(gray_frame_colored, box_2, colors[1])
     
    #resize to output resolution
    resized_frame = cv2.resize(gray_frame_colored, output_resolution)
    return resized_frame


def get_avg_hsv_from_roi(frame,roi):
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert to HSV
    roi_hsv = hsv_image[roi[1]:roi[1]+roi[2], roi[0]:roi[0]+roi[3]]
    mean_hsv = np.mean(roi_hsv, axis=(0, 1))
    return mean_hsv

def construct_filter(hsv_target, h_margin = 10, s_margin = 50, v_margin = 50):
    h_min = np.clip(hsv_target[0]  - h_margin, 0,179)
    h_max = np.clip(hsv_target[0]  + h_margin, 0,179)
    
    s_min  = np.clip(hsv_target[1] - s_margin , 0, 255)
    s_max  = np.clip(hsv_target[1] + s_margin , 0, 255)
    
    v_min  = np.clip(hsv_target[2] - v_margin , 0, 255)
    v_max  = np.clip(hsv_target[2] + v_margin , 0, 255)

    
    lower_color = np.array([h_min, s_min, v_min])  
    upper_color = np.array([h_max, s_max, v_max])
    
    return [lower_color, upper_color]

def detect_color(frame,color_bounds,min_size =50):
    #extract bounds 
    lower_color = color_bounds[0]
    upper_color = color_bounds[1]
    
    #convert image to hue, saturation and visability
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Mask the specific color
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)
    
    #default value
    bounding_rect = 0,0,50,50
    # Find contours of the masked area
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0,0,False,bounding_rect

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < min_size:  # Minimum size to filter noise
        return 0,0,False,bounding_rect
    
    # Get the centroid of the largest contour
    M = cv2.moments(largest_contour)
    
    if M['m00'] == 0: #0 area case
        return 0,0,False,bounding_rect
        
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    bounding_rect = cv2.boundingRect(largest_contour)
    return cx, cy, True, bounding_rect

# Function to draw a crosshair
def draw_crosshair(img, center, size=20, color=(0, 0, 255), thickness=2):
    x, y = center
    # Horizontal line (x-direction)
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    # Vertical line (y-direction)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)

