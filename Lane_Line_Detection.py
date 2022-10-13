import cv2
import numpy as np

#Select Camera 
cap = cv2.VideoCapture(0)

while True:
    
    #Call Camera
    _ , frame = cap.read()
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Region restriction 
    def region_of_interest (frame):
        height = frame.shape[0]
        triangle = np.array([[(200, height), (1100, height), (550,250)]])
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, triangle, 250)
        mask_frame = cv2.bitwise_and(frame, mask) 
        return mask_frame

    #White color array
    lower_white = np.array([0,0,255])
    high_white = np.array([255,0,255])

    #White Mask 
    white_mask = cv2.inRange(hsv_frame, lower_white, high_white)
    white = cv2.bitwise_and(frame, frame, mask=white_mask)

    #Canny edge detector
    edges = cv2.Canny(white, 100, 250)
    cropped_frame = region_of_interest(edges)
    #Hough Lines
    lines = cv2.HoughLinesP (cropped_frame, 1 , np.pi/180,50, maxLineGap = 75)

    #Lines 
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
        
     
    #cropped_frame2 = region_of_interest(frame)
    cv2.imshow("frame", frame) 

   #cv2.imshow("white", white)
    cv2.imshow("edges", cropped_frame)
    

    key = cv2.waitKey(1)
    if key == 27:
        break