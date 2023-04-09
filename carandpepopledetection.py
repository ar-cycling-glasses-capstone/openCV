import cv2
#from imutils import paths
import numpy as np
#import imutils
import schedule 
import time 


pollingrate = 2
carsFront_cascade = cv2.CascadeClassifier('models/cars2.xml')
carsBack_cascade = cv2.CascadeClassifier('models/cars.xml')
body_cascade = cv2.CascadeClassifier('models/fullbody.xml')
people_cascade = cv2.CascadeClassifier('models/pedestrian.xml')
distanceCutoff = 100

def detect_cars_and_pedestrain(frame):
    print("job run")
    cars1 = carsFront_cascade.detectMultiScale(frame, 1.15, 40)
    cars2 = carsBack_cascade.detectMultiScale(frame, 1.15, 40)
    people =  people_cascade.detectMultiScale(frame, 1.15, 40)
    pedistrain = body_cascade.detectMultiScale(frame, 1.15, 40)
    
    carDetected= False
    peopleDetected = False
    peoples = 0
    cars = 0
    result = []
    for (x, y, w, h) in cars1: 
        
        #print(x, y, w, h)
        
        if y<distanceCutoff:
            carDetected = True
            cars+=1
        #     cv2.rectangle(frame, (x+1, y+1), (x+w,y+h), color=(275,7,75), thickness=5)
        # else:
        #     cv2.rectangle(frame, (x+1, y+1), (x+w,y+h), color=(275,7,75), thickness=2)
    for (x, y, w, h) in cars2: 
        
        
        
        if y<distanceCutoff:
            carDetected=True
            cars+=1
            #cv2.rectangle(frame, (x+1, y+1), (x+w,y+h), color=(217, 255, 0), thickness=5)
            #else:
            #cv2.rectangle(frame, (x+1, y+1), (x+w,y+h), color=(217, 255, 0), thickness=2)       
    for(x, y, w, h) in pedistrain:
        peopleDetected = True
        peoples+=1
        #cv2.rectangle(frame, (x, y), (x+w, y+h), color=(0, 255, 255), thickness=2) 
    for(x, y, w, h) in people:
        peopleDetected = True
        peoples+=1
        #cv2.rectangle(frame, (x, y), (x+w, y+h), color=(0, 255, 255), thickness=2) 
    return peopleDetected, peoples, carDetected, cars, frame
   

def distance_to_camera(knownWidth, focalLength, perWidth):
	    # compute and return the distance from the maker to the camera
	    return (knownWidth * focalLength) / perWidth


def Simulator():
    #CarVideo = cv2.VideoCapture('cars.mp4')
    #take input from Camera
    CarVideo = cv2.VideoCapture(0)
    
    while CarVideo.isOpened():
        ret,frame = CarVideo.read()
        controlkey = cv2.waitKey(1)
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#conv to grayscale        
            peopleDetected, peoples, carDetected, cars, cars_frame = detect_cars_and_pedestrain(frame)
            schedule.every(pollingrate).seconds.do(detect_cars_and_pedestrain, gray)
            cv2.imshow('frame',cars_frame)
            
            
            
            time.sleep(1)
        else:
            break
        if controlkey == ord('q'):
            break
        schedule.run_pending()
        

    CarVideo.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    Simulator()
