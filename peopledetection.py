import cv2


scale_factor = 1.2
min_neighbors = 3
min_size = (50, 50)
# Load the cascade
body_class=cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # swap out with wwhatever being tracked

# To capture video from webcam. 
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read() #get frames

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#conv to grayscale for faster processing 
    
    #detect bodies
    bodies = body_class.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,minSize=min_size)
    # if at least 1 bod detected
    if len(bodies) >= 0:
        # Draw a rectangle around it
        for (x, y, w, h) in bodies:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Display the resulting frame
        
    cv2.imshow("face detection", frame)
    
    if cv2.waitKey(1)& 0xFF == ord('q'): #press q on keyboard to quit
        break
cap.release()
cv2.destroyAllWindows()
cv2