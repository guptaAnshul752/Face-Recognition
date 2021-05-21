# Face Recognition Python Script using KNN(K-Nearest Neighbor Algorithm)


# 1. load the training data (numpy arrays of all the persons)
		# x-values are stored in the numpy arrays
		# y-values we need to assign for each person
# 2. Read a video stream using opencv
# 3. extract faces out of it
# 4. use knn to find the prediction of face (int)
# 5. map the predicted id to name of the user 
# 6. Display the predictions on the screen - bounding box and name

import cv2
import numpy as np
import os

# KNN algorithm

#Euclidean Distance
def distance(x1,x2):
	return np.sqrt(sum(x1-x2)**2)

#KNN
def knn(train,test,k=5):

	dist = []
	
	for i in range(train.shape[0]):

		# Get the vector and label from training data
		ix = train[i, :-1]
		iy = train[i, -1]

		# Compute the distance from test point
		d = distance(test, ix)

		dist.append([d, iy])

	# Sort based on distance and get top k
	dk = sorted(dist, key=lambda x: x[0])[:k]

	# Retrieve only the labels
	labels = np.array(dk)[:, -1]
	
	# Get frequencies of each label
	output = np.unique(labels, return_counts=True)

	# Find max frequency and corresponding label
	index = np.argmax(output[1])

	return output[0][index]

#Capture using default Webcam
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
#eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
#smileCascade = cv2.CascadeClassifier("haarcascade_smile.xml")

skip = 0

# Array to store face data
face_data = []

# Dataset Path
dataset_path = './imagedata/' 

# Labels
label = []

class_id = 0; #Labels for the given file

names = {}


# Data Preparations (Loading Train data files)
for fx in os.listdir(dataset_path):

	if fx.endswith('.npy'):

		# Create mapping with class_id and names
		names[class_id] = fx[:-4]

		data_item = np.load(dataset_path+fx,allow_pickle=True)

		face_data.append(data_item)

		# Giving data an corresponding appropiate label
		target = class_id*np.zeros((data_item.shape[0],))

		class_id += 1

		label.append(target)

#Concatenating data & label into a single list (List of List)

face_dataset = np.concatenate(face_data,axis=0) ## X-train

face_label = np.concatenate(label,axis=0).reshape((-1,1)) ## Y-train

# -1 in reshaping means that python itself would suggest the appropiate value for dimensions

print(face_dataset.shape)

print(face_label.shape)

# Concatenate both face_dataset & face_label

# Coz Knn accepts 1 training matrix only containing both xtrain and ytrain
trainset = np.concatenate((face_dataset,face_label),axis = 1)

print(trainset.shape) # 30000 features and 1 label i.e (x,30001)

# Test-Data
while True:

	ret,frame = cap.read()
	#gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	if ret == False:
		continue

	# Detecting Face
	# Parameters --> Frame, Scaling factor, Number of Neighbors
	faces = face_cascade.detectMultiScale(frame,1.15,5)
	#print('Shape of Face before: ' , faces)
	#faces1 = face_cascade.detectMultiScale(gray_frame,1.15,5)


	# Detecting Smile
	#smile = smileCascade.detectMultiScale(gray_frame,1.3,5)

	# Detecting Eyes
	#eyes = eyeCascade.detectMultiScale(gray_frame,1.3,5)
	#face_section = 0
	# Iterate from last position as last face needs to be picked(last face being largest(Area))
	for (x,y,w,h) in faces:

		# Extract (CropOut) required face : Region of Interest
		# pixels
		offset = 10 # Padding
		# Increasing face selection Area
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		# Resizing
		face_section = cv2.resize(face_section,(100,100))

		# Predicted label
		pred = knn(trainset,face_section.flatten())

		# Display on Screen ,Name and rectangle around it
		pred_name = names[int(pred)]
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

	#Display
	cv2.imshow('Faces', frame)
	#cv2.imshow('Face Section',face_section)
	#cv2.imshow('GrayFrame', gray_frame)

	#Wait for user input q, then loop will stop (video also)
	#Converting 32-bit integer into 8 bit using BitWise AND ,for comparision with ASCII value of q
	key_pressed = cv2.waitKey(1) & 0xFF  ## Waitkey(1) program will wait for 1 millisecond before next iteration

	if key_pressed == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
