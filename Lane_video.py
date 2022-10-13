from symbol import parameters
import cv2
import numpy as np

def make_coordinates (image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def avarage_slope_intercept (image,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters [0]
        intercept = parameters [1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis = 0)
    rigth_fit_average = np.average(right_fit, axis = 0)
    left_line = make_coordinates (image, left_fit_average)
    right_line = make_coordinates (image, rigth_fit_average)
    return np.array([left_line, right_line])


#Canny function
def canny(image):
#Create a gray scale on the image 
    gray =cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#Applying a gaussianblur on a grayscale image on a 5 by 5 kernel
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

#Canny function traces the edge itensity with respect to other pixels
    canny = cv2.Canny(blur, 50, 150)

    return canny

#Define a function to display the lines
def display_lines(image, lines):
    line_image = np.zeros_like (image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255,0,0), 10)
            
    return line_image


#Region of interest
def region_of_interest (image):
    height = image.shape[0]
    polygons = np.array([[(200,height), (1100,height), (575,250)]])
    mask =np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask) #
    return masked_image

cap= cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()

#Call Canny on the lane_image
    canny_image = canny(frame)
    cropped_image = region_of_interest (canny_image)

#Hough Transform 
    lines = cv2.HoughLinesP (cropped_image, 2, np.pi/180, 100, np.array([]), maxLineGap=10)
    avaraged_lines = avarage_slope_intercept (frame, lines)
    line_image = display_lines (frame, avaraged_lines)

#Blend the line_image to the original image
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

#Show Result
    cv2.imshow("result", combo_image)
#End the program
    if cv2.waitKey(1) & 0xFF== ord('q'): 
        break
    
cap.release()
cv2.destroyAllWindows()