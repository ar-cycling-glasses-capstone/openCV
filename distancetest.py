import cv2


scale_factor = 1.2
min_neighbors = 3
min_size = (50, 50)
# Load the cascade
body_class=cv2.CascadeClassifier("models/palm.xml") # swap out with wwhatever being tracked

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# fontScale
fontScale = 1
# font
font = cv2.FONT_HERSHEY_SIMPLEX
# Red color in BGR
color = (0, 0, 255)
  
# Line thickness of 2 px
thickness = 2
ratio = 10/(10*124)
  
scale = 2
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
            cv2.putText(frame,("Distance [cm] = "+str(int(w*h*ratio))), (x,y),font, fontScale,
                  color, thickness, cv2.LINE_AA, True) #Draw the text
            
            
            
    # Display the resulting frame
        
    cv2.imshow("face detection", frame)
    
    if cv2.waitKey(1)& 0xFF == ord('q'): #press q on keyboard to quit
        break
cap.release()
cv2.destroyAllWindows()
cv2
