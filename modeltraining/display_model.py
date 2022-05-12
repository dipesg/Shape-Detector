from fileinput import filename
import cv2, numpy as np, pickle
from shapedetection.utils import Helper

img_size = 64 
#import model
filename = '../trainedmodel/model/finalized_model.sav'
model = pickle.load(open(filename, 'rb'))
dimData = np.prod([img_size, img_size])

cap = cv2.VideoCapture(0)
while True:
    #read image from video, create a copy to draw on
    _, img= cap.read()
    imgc = img.copy()
    height, width, _ = img.shape

    #mask of the green regions in the image  
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  
    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, mask = Helper().only_color(img)

    #find the contours in the image
    contours, _= cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #iterate through the contours, "Keras, what shape is this contour?"
    for c in contours:
        #if the contour is too big or too small, it can be ignored
        area = cv2.contourArea(c)
        #print area
        if area> 3000 and area< 1180000:

            #crop out the green shape
            roi, coords = Helper().bbox(img, c)

            #filter out contours that are long and stringy
            if np.prod(roi.shape[:2])>10:

                #get the black and white image of the shape
                roi = cv2.resize(roi, (img_size, img_size))
                _, roi = Helper().only_color(roi)
                roi= 255-roi #Keras likes things black on white
                mask = cv2.resize(roi, img_size, img_size)
               # mask = mask.reshape(dimData)
                #mask = mask.astype('float32')
                mask /=255
                image = np.expand_dims(mask, axis=0)
                #feed image into model
                prediction = model.predict(image)[0].tolist()  #.reshape(1,dimData)

                #create text --> go from categorical labels to the word for the shape.
                text = ''
                p_val, th = .25, .5
                if max(prediction)> p_val:
                    if prediction[0]>p_val and prediction[0]==max(prediction): text, th =  'triangle', prediction[0]
                    if prediction[1]>p_val and prediction[1]==max(prediction): text, th =  'star', prediction[1]
                    if prediction[2]>p_val and prediction[2]==max(prediction): text, th =  'square', prediction[2]
                    if prediction[3]>p_val and prediction[3]==max(prediction): text, th =  'circle', prediction[3]
                
                #draw the contour
                cv2.drawContours(imgc, c, -1, (0,0,255), 1)

                #draw the text
                org, font, color = (coords[0], coords[1]+int(area/400)), cv2.FONT_HERSHEY_SIMPLEX, (0,0,255)
                cv2.putText(imgc, text, org, font, int(2.2*area/15000), color, int(6*th), cv2.LINE_AA)

                #paste the black and white image onto the source image (picture in picture)
                if text!='': imgc[imgc.shape[0]-200:imgc.shape[0], img.shape[1]-200:img.shape[1]] = cv2.cvtColor(cv2.resize(roi, (200,200)), cv2.COLOR_GRAY2BGR)
                
    cv2.imshow('img', cv2.resize(imgc, (640, 480))) #expect 2 frames per second
    k = cv2.waitKey(100)
    if k == 27: break
    
cap.release()
cv2.destroyAllWindows()

