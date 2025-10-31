#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 15:49:25 2023

@author: jiangdawei
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

lower_blue = np.array([90, 50, 50])
upper_blue = np.array([120, 255, 255])
lower_red = np.array([0, 50, 50])
upper_red = np.array([180, 255, 255])
lower_green = np.array([50, 50, 50])
upper_green = np.array([80, 255, 255])

# Load video
cap = cv2.VideoCapture('/Users/jiangdawei/Downloads/低頻.mov')
# 设置开始时间（毫秒）
start_time = 1000
cap.set(cv2.CAP_PROP_POS_MSEC, start_time)

# 设置结束时间（毫秒）
end_time = 31000
frame_counter1 = 0
# List to store the trajectory of the object
trajectory1X = []
trajectory1Y = []
trajectory2X = []
trajectory2Y = []
theta1 = []
theta2 = []
velocity1 = []
velocity2 = []
ag1 = []
ag2 = []
# Variable to store the initial coordinates of the blue object
initial_coordinates1 = None
initial_coordinates2 = None
initial_coordinates3 = None
# Initialize variables
initial_angle = 0.0
turn = 0
prev_angle = None  # Initialize prev_angle
agv1 = 0
# Initialize a counter to keep track of the number of frames processed
frame_counter = 0
# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)
#print(fps)
# Calculate the time interval between frames in milliseconds
interval = 1 / fps

while(cap.isOpened()):
    ret, frame = cap.read() #ret is a boolean value that indicates if the frame was successfully read
    if not ret:  #video is ended then it breaks
        break
    
    # 获取当前视频位置
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
    
    # 如果当前时间超过了结束时间，就退出循环
    if current_time >= end_time:
        break
    # Apply Gaussian blur
    blurred_frame = cv2.GaussianBlur(frame, (3, 3), 0)
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    
    # Threshold the HSV image to get only blue color
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    #mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    # Bitwise-AND mask and original image
    res_blue = cv2.bitwise_and(blurred_frame, blurred_frame, mask=mask_blue)
    #res_red = cv2.bitwise_and(blurred_frame, blurred_frame, mask=mask_red)
    res_green = cv2.bitwise_and(blurred_frame, blurred_frame, mask=mask_green)
    # Convert the result to grayscale
    gray_res_blue = cv2.cvtColor(res_blue, cv2.COLOR_BGR2GRAY)
    #gray_res_red = cv2.cvtColor(res_red, cv2.COLOR_BGR2GRAY)
    gray_res_green = cv2.cvtColor(res_green, cv2.COLOR_BGR2GRAY)
    # Find contours in the grayscale image
    contours_blue, hierarchy_blue = cv2.findContours(gray_res_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours_red, hierarchy_red = cv2.findContours(gray_res_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, hierarchy_green = cv2.findContours(gray_res_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours_red) > 0:
        largest_contour_red = max(contours_red, key=cv2.contourArea)
        
        # Find the center of the largest red contour
        M_red = cv2.moments(largest_contour_red)
        if M_red["m00"] != 0:
            cX_red = int(M_red["m10"] / M_red["m00"])
            cY_red = int(M_red["m01"] / M_red["m00"])
            
            if initial_coordinates1 is None:
                initial_coordinates1 = (cX_red, cY_red)
                
    # Calculate the relative coordinates of the center
    relative_cX_red = (np.loadtxt("/Users/jiangdawei/Downloads/X_data2.txt"))
    relative_cY_red = (np.loadtxt("/Users/jiangdawei/Downloads/Y_data2.txt"))
    #print(len(relative_cX_red))
    # Check if frame_counter is within the bounds of the loaded data
    if frame_counter1 < len(relative_cX_red):
        # Extract coordinates for the current frame
        current_relative_cX_red = int(relative_cX_red[frame_counter])
        current_relative_cY_red = int(relative_cY_red[frame_counter])

        # Draw a circle using the current coordinates
        cv2.circle(frame, (current_relative_cX_red, current_relative_cY_red), 20, (0, 0, 255), -1)
        
    if len(contours_green) > 0:
        largest_contour_green = max(contours_green, key=cv2.contourArea)
        
        # Find the center of the largest contour
        M = cv2.moments(largest_contour_green)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if initial_coordinates2 is None:
                initial_coordinates2 = (cX, cY)
            cv2.circle(frame, (cX, cY), 20, (0, 255, 0), -1)
            trajectory1X.append(cX)
            trajectory1Y.append(cY)
            
            
            # Calculate the relative coordinates of the center
            relative_cX_green = cX - relative_cX_red
            relative_cY_green = cY - relative_cY_red
            
            # Store the position of the center
            trajectory1.append((relative_cX_green, relative_cY_green))
            
            # Print the coordinates of the center of the object
            #print(len(relative_cX_green))
            flip_1 = 0
            # Draw a circle at the center of the largest contour
            cv2.circle(frame, (cX, cY), 20, (0, 255, 0), -1)
            angle1 = np.arctan2(relative_cY_green,relative_cX_green)-1.57
            if len(ag1) > 0:
                if (angle1 + flip_1 < ag1[-1] - 3.14).any():
                    flip_1 += 6.28
                elif (angle1 + flip_1 > ag1[-1] + 3.14).any():
                    flip_1 -= 6.28
            ag1.append(angle1 + flip_1)  
            

        # Find the largest contour
    if len(contours_blue) > 0:
        largest_contour_blue = max(contours_blue, key=cv2.contourArea)
        
        # Find the center of the largest contour
        M = cv2.moments(largest_contour_blue)
        if M["m00"] != 0:
            cX2 = int(M["m10"] / M["m00"])
            cY2 = int(M["m01"] / M["m00"])
            if initial_coordinates3 is None:
                initial_coordinates3 = (cX2, cY2)
            trajectory2X.append(cX2)
            trajectory2Y.append(cY2)
            cv2.circle(frame, (cX2, cY2), 20, (255, 0, 0), -1)
            
            # Calculate the relative coordinates of the center
            relative_cX_blue = cX2 - cX
            relative_cY_blue = cY2 - cY
            #print(relative_cX,relative_cY)
            # Store the position of the center
            trajectory2.append((relative_cX_blue, relative_cY_blue))
            
            # Print the coordinates of the center of the object
            #print(cX2,cY2)
            flip_2 = 0
            # Draw a circle at the center of the largest contour
            cv2.circle(frame, (cX2, cY2), 20, (255, 0, 0), -1)
            angle2 = np.arctan2(relative_cY_blue, relative_cX_blue) - 1.57
            if len(ag2) > 0:
                if (angle2 + flip_2 < ag2[-1] - 3.14).any():
                    flip_2 += 6.28
                elif (angle2 + flip_2 > ag2[-1] + 3.14).any():
                    flip_2 -= 6.28
            ag2.append(angle2 + flip_2)
        #print(theta2)
        
        frame_counter += 1
        # Display the resulting frames
        cv2.imshow('frame', frame)
    # Find the largest contour
    print(len(trajectory1X))
    np.savetxt("data1x1.txt",trajectory1X)
    np.savetxt("data1y1.txt",trajectory1Y)
    np.savetxt("data2x1.txt",trajectory2X)
    np.savetxt("data2y1.txt",trajectory2Y)
    times = np.linspace(0, (frame_counter) * interval, frame_counter,endpoint=False)
    np.savetxt("time_medium2.txt",times)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Check if trajectory is empty before trying to plot
if len(trajectory1) > 0:
    trajectory1 = np.array(trajectory1)
    # Create a time array based on the number of frames processed
    times = np.linspace(0, (frame_counter) * interval, frame_counter,endpoint=False)
    #print(frame_counter)
    for i in range(len(ag1)-1):
        omega1 = (ag1[i+1]-ag1[i])/(times[1]-times[0])
        velocity1.append(omega1)
    np.savetxt("theta1_medium.txt",ag1)
    np.savetxt("omega1_medium.txt",velocity1)
# Check if trajectory is empty before trying to plot
if len(trajectory2) > 0:
    trajectory = np.array(trajectory2)
    # Create a time array based on the number of frames processed
    times = np.linspace(0, frame_counter * interval, frame_counter, endpoint=False)
    #print(frame_counter)
    for i in range(len(ag2)-1):
        omega2 = (ag2[i+1]-ag2[i])/(times[1]-times[0])
        velocity2.append(omega2)
 
    x = trajectory[:,0]
    y = trajectory[:,1]
    np.savetxt("theta2_medium.txt",ag2)
    np.savetxt("time_medium.txt",times)
    np.savetxt("omega2_medium.txt",velocity2)

    
    # Plot the trajectory
    plt.figure(figsize=(6, 6))
    plt.plot(times, ag2)
    plt.title('time-theta2')
    plt.xlabel('time')
    plt.ylabel('theta2')
    plt.show()
    
    # Plot the trajectory
    plt.figure(figsize=(6, 6))
    plt.plot(times, ag1)
    plt.title('time-theta1')
    plt.xlabel('time')
    plt.ylabel('theta1')
    plt.show()
   
    # Compute the FFT of theta2
    theta1_fft = np.fft.fft(ag1)
    # Compute the power spectral density (PSD)
    psd1 = np.abs(theta1_fft/len(times))**2
    # Compute the frequencies corresponding to the FFT
    freqs = np.fft.fftfreq(len(ag1), d=0.001)
    # Plot the PSD
    plt.figure(figsize=(6, 6))
    plt.plot(freqs, psd1)
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.show()
    
    # Compute the FFT of theta2
    theta2_fft = np.fft.fft(ag2)
    # Compute the power spectral density (PSD)
    psd = np.abs(theta2_fft/len(times))**2
    # Compute the frequencies corresponding to the FFT
    freqs = np.fft.fftfreq(len(ag2), d=0.001)
    # Plot the PSD
    plt.figure(figsize=(6, 6))
    plt.plot(freqs, psd)
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.show()

# Release the capture
cap.release()
cv2.destroyAllWindows()

