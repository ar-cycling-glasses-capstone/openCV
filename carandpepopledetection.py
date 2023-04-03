import cv2
#from imutils import paths
import numpy as np
#import imutils


carsFront_cascade = cv2.CascadeClassifier('models/cars2.xml')
carsBack_cascade = cv2.CascadeClassifier('models/cars.xml')
body_cascade = cv2.CascadeClassifier('models/fullbody.xml')
people_cascade = cv2.CascadeClassifier('models/pedestrian.xml')
distanceCutoff = 100

def detect_cars_and_pedestrain(frame):
    cars1 = carsFront_cascade.detectMultiScale(frame, 1.15, 40)
    cars2 = carsBack_cascade.detectMultiScale(frame, 1.15, 40)
    people =  people_cascade.detectMultiScale(frame, 1.15, 40)
    pedistrain = body_cascade.detectMultiScale(frame, 1.15, 40)
    for (x, y, w, h) in cars1: 
        
        #print(x, y, w, h)
        
        if y<distanceCutoff:
            
            cv2.rectangle(frame, (x+1, y+1), (x+w,y+h), color=(275,7,75), thickness=5)
        else:
            cv2.rectangle(frame, (x+1, y+1), (x+w,y+h), color=(275,7,75), thickness=2)
    for (x, y, w, h) in cars2: 
        
        print(x, y, w, h)
        
        if y<distanceCutoff:
            
            cv2.rectangle(frame, (x+1, y+1), (x+w,y+h), color=(217, 255, 0), thickness=5)
        else:
            cv2.rectangle(frame, (x+1, y+1), (x+w,y+h), color=(217, 255, 0), thickness=2)       
    for(x, y, w, h) in pedistrain:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color=(0, 255, 255), thickness=2) 
    for(x, y, w, h) in people:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color=(0, 255, 255), thickness=2) 
    return frame


def distance_to_camera(knownWidth, focalLength, perWidth):
	    # compute and return the distance from the maker to the camera
	    return (knownWidth * focalLength) / perWidth


def Simulator():
    CarVideo = cv2.VideoCapture('cars.mp4')
    #take input from Camera
    #CarVideo = cv2.VideoCapture(-1)
    
    while CarVideo.isOpened():
        ret, frame = CarVideo.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#conv to grayscale for faster processing 
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 35, 125)
        controlkey = cv2.waitKey(1)
        # find the contours in the edged image and keep the largest one;
        # we'll assume that this is our piece of paper in the image
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key = cv2.contourArea)
        # compute the bounding box of the of the paper region and return it
        print(cv2.minAreaRect(c))
        if ret:        
            cars_frame = detect_cars_and_pedestrain(frame)
            cv2.imshow('frame', cars_frame)
        else:
            break
        if controlkey == ord('q'):
            break

    CarVideo.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    Simulator()
