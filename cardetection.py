import cv2


#load cascade
cars_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

scaleFactor = 1.05 #how much img sized is reduced 
minNeighbors = 3#how many neighbors each candidate rectangle should have in it



def detect_cars(frame):
    cars = cars_cascade.detectMultiScale(frame, 1.15, 4)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w,y+h), color=(0, 255, 0), thickness=2)
    return frame

def Simulator():
    CarVideo = cv2.VideoCapture('cars.mp4')#cap from test video
   
    #cap = cv2.VideoCapture(0) #cap from webcam
    while CarVideo.isOpened():
        ret, frame = CarVideo.read()
        controlkey = cv2.waitKey(1)
        if ret:        
            cars_frame = detect_cars(frame)
            cv2.imshow('frame', cars_frame)
        else:
            break
        if controlkey == ord('q'):
            break
​
    CarVideo.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    Simulator()