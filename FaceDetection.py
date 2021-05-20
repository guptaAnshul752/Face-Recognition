# Face Detection Python Script

# Video Capture using OpenCv

# Importing OpenCV and Numpy
# OpenCV to capture the images and videostream
# Numpy to store face data

import cv2
import numpy as np

#Capture using default Webcam
cap = cv2.VideoCapture(0)

# Face Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

# Eye Cascade
#eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# Smile Cascade
#smileCascade = cv2.CascadeClassifier("haarcascade_smile.xml")

skip = 0

# Array to store face data
face_data = []

# Storage location
datapath = './imagedata/'

# Name of Person
file_name = input('Enter your name: ')

while True:

	ret,frame = cap.read()

	# Grayscale frame
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	if ret == False:
		continue

	# Detecting Face
	# Parameters --> Frame, Scaling factor, Number of Neighbors
	faces = face_cascade.detectMultiScale(frame,1.15,5)

	#print('Shape of Face before: ' , faces)
	#grayscale cascade
	faces1 = face_cascade.detectMultiScale(gray_frame,1.15,5)

	# Store the face having maximum area(w,h)(w*h)
	faces = sorted(faces , key = lambda f:f[2]*f[3])

	# Detecting Smile
	#smile = smileCascade.detectMultiScale(gray_frame,1.3,5)

	# Detecting Eyes
	#eyes = eyeCascade.detectMultiScale(gray_frame,1.3,5)

	# Face Section
	face_section = 0

	# Iterate from last position as last face needs to be picked(last face being largest(Area))
	for (x,y,w,h) in faces[-1:]:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #2 means thickness

		# Extract (CropOut) required face : Region of Interest
		# pixels
		offset = 10 # Padding
		# Increasing face selection Area
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		# Resizing
		face_section = cv2.resize(face_section,((100,100)))

		# Store every 10th face
		if (skip % 10 == 0):
			face_data.append(face_section)
			print(len(face_data))

		skip += 1
		
	#for (x,y,w,h) in faces1:
		#cv2.rectangle(gray_frame,(x,y),(x+w,y+h),(255,0,0),2)

	#for (ex,ey,ew,eh) in eyes:
	#	cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
		#cv2.putText(frame,'Eye',(x + ex,y + ey), 1, 1, (0, 255, 0), 1)

	#for (sx, sy, sw, sh) in smile:
	#	cv2.rectangle(frame, (sh,sy), (sx+sw, sy+sh), (0, 0, 255),2)
		#cv2.putText(frame,'Smile',(x + sx,y + sy), 1, 1, (0, 255, 0), 1)

	#Display
	cv2.imshow('Video', frame)

	cv2.imshow('Face Section',face_section)

	#cv2.imshow('GrayFrame', gray_frame)

	#Wait for user input q, then loop will stop (video also)
	#Converting 32-bit integer into 8 bit using BitWise AND ,for comparision with ASCII value of q
	key_pressed = cv2.waitKey(1) & 0xFF  ## Waitkey(1) program will wait for 1 millisecond before next iteration

	if key_pressed == ord('q'):
		break

# Convert face_data list into numpy array
face_data = np.asarray(face_data)

face_data = face_data.reshape((face_data.shape[0],-1))

print(face_data.shape)

# Save File
np.save(datapath + file_name + '.npy',face_data)

print('Data succesfully saved at' + datapath + file_name + '.npy')

cap.release()

cv2.destroyAllWindows()


'''
Here is a list of the most common parameters of the detectMultiScale function :
scaleFactor : Parameter specifying how much the image size is reduced at each image scale.
minNeighbors : Parameter specifying how many neighbors each candidate rectangle should have to retain it.
minSize : Minimum possible object size. Objects smaller than that are ignored.
maxSize : Maximum possible object size. Objects larger than that are ignored.
'''
# The Scaling Factor basically depicts the factor through which the image size would be reduced when passed through the Classifier. 
#If we specify a factor 1.x (say) then our image would be resized by x%. We need to have this factor because the 
#frontal-face classifier is trained on Multi Layer Perceptrons which do not work well if the testing image
# is too different wrt the scale as compared to the training ones (which they must’ve done at their end).