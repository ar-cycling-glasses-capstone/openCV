# import the opencv library
import cv2

from imutils import paths
import numpy as np
import imutils


hand_cascade = cv2.CascadeClassifier('fist.xml')

# define a video capture object
vid = cv2.VideoCapture(-1)

scale = 2
def detect_fist(frame):
	fist = hand_cascade.detectMultiScale(frame, 1, 400)
	for (x, y, w, h) in fist:
		cv2.rectangle(frame, (x+1, y+1), (x+w,y+h), color=(275,7,75), thickness=5)
		dist = scale *(x/y)
		cv2.PutText(frame,("Distance = "+str(dist)), (x,y),font, 255) #Draw the text
      
      
while(True):
	
	# Capture the video frame
	# by frame
	ret, frame = vid.read()

	# Display the resulting frame
	cv2.imshow('frame', frame)
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
		hand_frame = detect_fist(frame)
		cv2.imshow('frame', hand_frame)
	else:
		break
	if controlkey == ord('q'):
		break
	# the 'q' button is set as the
	# quitting button you may use any
	# desired button of your choice
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
