#!/usr/bin/env python2
#
# Example to classify faces.
# Brandon Amos
# 2015/10/11
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import time
import datetime
start = time.time()

import argparse
import os
import pickle
from operator import itemgetter
import numpy as np
np.set_printoptions(precision=2)
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import sys
import dlib
import cv2
import serial
import openface
from skimage import io
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import serial
import pymysql

db = pymysql.connect(host='localhost',user='phpmyadmin',passwd='badar', db='fr_database')
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
###################### CARD SCANNING ####################################


	## whether or not the arduino is connected
	connected = False

	## open the serial port that your ardiono 
	## is connected to.
	ser = serial.Serial("/dev/ttyUSB0", 9600)
	serin = []
	## loop until the arduino tells us it is ready

	while not connected:
		connected = True

	## Wait until the arduino tells us it 
	## is finished blinking
	#while connected:
	print("Please Scan Card: ")
	scr = ser.read(11)
	## database work
	print(scr)
	cursor = db.cursor(pymysql.cursors.DictCursor)
	query = ("SELECT type FROM cards WHERE card_id=%s")
	result=cursor.execute(query, (scr))
	result_set = cursor.fetchall ()
	for row in result_set:
		type = row["type"]
		if type== "person":
			query = ("SELECT reg_id,status FROM person WHERE card_id=%s")
			result=cursor.execute(query, (scr))
			if result == 1:
				result_set = cursor.fetchall ()
				for row in result_set:
					reg_id = row["reg_id"]
					status = row["status"]
		
					if(status == 0):
				
						print("{} is checked-In at {}".format(reg_id,datetime.datetime.now()))
						time=datetime.datetime.now()
						cursor.execute("""INSERT INTO checkin_person (personid,time) VALUES (%s,%s)""",(reg_id,time))
   						db.commit()
						cursor.execute("""UPDATE person SET status=%s where reg_id=%s""",(1,reg_id))
						db.commit()

					elif (status == 1):
				
						print("{} is checked-Out at {}".format(reg_id,datetime.datetime.now()))
						time=datetime.datetime.now()
						cursor.execute("""INSERT INTO checkout_person (personid,time) VALUES (%s,%s)""",(reg_id,time))
   						db.commit()
						cursor.execute("""UPDATE person SET status=%s where reg_id=%s""",(0,reg_id))
						db.commit()
						scan_card()

				
			elif result == 0:
				print ("Invalid Card. Please visit Administrator office.\n")
				scan_card()
		elif type == "visitor":
			query = ("SELECT cnic,status FROM visitor WHERE card_id=%s")
			result=cursor.execute(query, (scr))
			if result == 1:
				result_set = cursor.fetchall ()
				for row in result_set:
					cnic = row["cnic"]
					status = row["status"]
		
					if(status == 0):
				
						print("{} is checked-In at {}".format(cnic,datetime.datetime.now()))
						time=datetime.datetime.now()
						cursor.execute("""INSERT INTO checkin_visitor (visitorid,time) VALUES (%s,%s)""",(cnic,time))
   						db.commit()
						cursor.execute("""UPDATE visitor SET status=%s where cnic=%s""",(1,cnic))
						db.commit()

					elif (status == 1):
				
						print("{} is checked-Out at {}".format(cnic,datetime.datetime.now()))
						time=datetime.datetime.now()
						cursor.execute("""INSERT INTO checkout_visitor (visitorid,time) VALUES (%s,%s)""",(cnic,time))
   						db.commit()
						cursor.execute("""UPDATE visitor SET status=%s where cnic=%s""",(0,cnic))
						db.commit()
					scan_card()
				
			elif result == 0:
				print ("Invalid Card. Please visit Administrator office.\n")
				scan_card()
						
	
	##---------------------------------------------------------------------------------------------
	## close the port and end the program
	ser.close()	

def camera():
	predictor_model = "shape_predictor_68_face_landmarks.dat"

	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--shape-predictor", required=True,
		help="path to facial landmark predictor", default=os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))
	ap.add_argument("-r", "--picamera", type=int, default=-1,
		help="whether or not the Raspberry Pi camera should be used")
	args = vars(ap.parse_args())

	
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	print("[INFO] loading facial landmark predictor...")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(args["shape_predictor"])

	# initialize the video stream and allow the cammera sensor to warmup
	print("[INFO] camera sensor warming up...")
	vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
	time.sleep(2.0)

	# loop over the frames from the video stream
	while True:
		# grab the frame from the threaded video stream, resize it to
		# have a maximum width of 400 pixels, and convert it to
		# grayscale
		image = vs.read()
		image = imutils.resize(image, width=1000)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
		# detect faces in the grayscale frame
		rects = detector(gray, 0)

		# loop over the face detections
		for rect in rects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
 
			# loop over the (x, y)-coordinates for the facial landmarks
			# and draw them on the image
			for (x, y) in shape:
				cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
	  
		# show the frame
		cv2.imshow("Frame", image)
		key = cv2.waitKey(1) & 0xFF
 
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break


	cv2.imwrite("captured_image.jpg",image)

	# Create a HOG face detector using the built-in dlib class
	face_detector = dlib.get_frontal_face_detector()
	face_pose_predictor = dlib.shape_predictor(predictor_model)
	face_aligner = openface.AlignDlib(predictor_model)
	win = dlib.image_window()
	win.set_image(image)


	# Run the HOG face detector on the image data
	detected_faces = face_detector(image, 1)	
	print("Found {} faces in the image file {}".format(len(detected_faces), "abc"))

	# Loop through each face we found in the image
	for i, face_rect in enumerate(detected_faces):

		# Detected faces are returned as an object with the coordinates 
		# of the top, left, right and bottom edges
		print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

		# Get the the face's pose
		pose_landmarks = face_pose_predictor(image, face_rect)
		#cv2.imwrite("aligned_face2.jpg", pose_landmarks)
		# Use openface to calculate and perform the face alignment
		alignedFace = face_aligner.align(534, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
		# Save the aligned image to a file
	cv2.imwrite("aligned_face.jpg", alignedFace)



os.system('clear')
camera()
