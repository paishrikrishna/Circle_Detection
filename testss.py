import cv2
import numpy as np

cap = cv2.VideoCapture('ballmotion.m4v')

fgbg2 = cv2.createBackgroundSubtractorMOG2(); 

while(1):
    _, frame = cap.read()
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([0,50,50])
    upper_red = np.array([6,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    contours, _ = cv2.findContours(mask0, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask0, contours, -1, (55,255,255),3)

    gray_blurred = cv2.blur(mask0, (3, 3)) 
      
    detected_circles = cv2.HoughCircles(gray_blurred,  
                       cv2.HOUGH_GRADIENT, 1, 120, param1 = 50, 
                   param2 = 25, minRadius = 40, maxRadius = 60) 
      
    if detected_circles is not None: 
      
        detected_circles = np.uint16(np.around(detected_circles)) 
      
        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2] 
            cv2.circle(frame, (a, b), 1, (0, 0, 0), 6) 


    cv2.imshow('frame',frame)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()