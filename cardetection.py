import cv2


carsFront_cascade = cv2.CascadeClassifier('cars2.xml')
carsBack_cascade = cv2.CascadeClassifier('cars.xml')
body_cascade = cv2.CascadeClassifier('fullbody.xml')
people_cascade = cv2.CascadeClassifier('pedestrian.xml')
distanceCutoff = 100

def detect_cars_and_pedestrain(frame):
    cars1 = carsFront_cascade.detectMultiScale(frame, 1.15, 4)
    cars2 = carsBack_cascade.detectMultiScale(frame, 1.15, 4)
    people =  people_cascade.detectMultiScale(frame, 1.15, 4)
    pedistrain = body_cascade.detectMultiScale(frame, 1.15, 4)
    for (x, y, w, h) in cars1: 
        
        #print(x, y, w, h)
        
        if y<distanceCutoff:
            
            cv2.rectangle(frame, (x+1, y+1), (x+w,y+h), color=(275,7,75), thickness=5)
        else:
            cv2.rectangle(frame, (x+1, y+1), (x+w,y+h), color=(275,7,75), thickness=2)
    for (x, y, w, h) in cars2: 
        
        #print(x, y, w, h)
        
        if y<distanceCutoff:
            
            cv2.rectangle(frame, (x+1, y+1), (x+w,y+h), color=(217, 255, 0), thickness=5)
        else:
            cv2.rectangle(frame, (x+1, y+1), (x+w,y+h), color=(217, 255, 0), thickness=2)       
    for(x, y, w, h) in pedistrain:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color=(0, 255, 255), thickness=2) 
    for(x, y, w, h) in people:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color=(0, 255, 255), thickness=2) 
    return frame

def Simulator():
    CarVideo = cv2.VideoCapture('cars.mp4')
    
    
    while CarVideo.isOpened():
        ret, frame = CarVideo.read()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#conv to grayscale for faster processing 
        controlkey = cv2.waitKey(1)
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
