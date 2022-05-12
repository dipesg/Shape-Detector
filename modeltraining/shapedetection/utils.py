import cv2
import numpy as np
class Helper:
    def __init__(self):
        pass
    
    def largest_contour(self, contours): 
        """
        finds the largest contour in a list of contours
        """
        return max(contours, key=cv2.contourArea)[1]
    
    def contour_center(self, c):
        """
        finds the center of a contour
        """
        M = cv2.moments(c)
        try: 
            center = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        except: 
            center = 0,0
        return center
    
    def only_color(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([98,50,50])
        upper = np.array([139,255,255])
        mask = cv2.inRange(hsv, lower, upper) 
        res = cv2.bitwise_and(img, img, mask=mask)
        #kernel = np.ones((3,3), np.uint)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return res, mask
    
    def bbox(self, img, c):
        pad = 60
        x,y,w,h = cv2.boundingRect(c)
        return img[y-pad:y+h+pad, x-pad:w+x+pad], (x,y)