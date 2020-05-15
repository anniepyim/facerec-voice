# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import face_recognition
from collections import Counter

from datetime import datetime

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s %(message)s')

# Setting paths
protoPath = os.path.join(os.path.dirname(__file__),
						 "../resources/face_detection_model/deploy.prototxt")
modelPath = os.path.join(os.path.dirname(__file__),
						 "../resources/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")
# embedderPath = os.path.join(os.path.dirname(__file__),
# 							 "../resources/face_embedding_model/openface_nn4.small2.v1.t7")
recognizerPath = os.path.join(os.path.dirname(__file__),
							  "../resources/face_detection_output/recognizer.pickle")
lePath = os.path.join(os.path.dirname(__file__),
					  "../resources/face_detection_output/le.pickle")


class RecognizerCam():

	def __init__(self):
		self.confidence = 0.6

	def run(self, confidence=0.6, run_time=5, warmup_time=1.0):

		self.confidence = confidence

		# load our serialized face detector from disk
		logger.info("Loading detector and recognizer")
		detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
		# detector.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

		# load our serialized face embedding model from disk
		# preferable target to MYRIAD
		# embedder = cv2.dnn.readNetFromTorch(embedderPath)
		# embedder.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

		# load the actual face recognition model along with the label encoder
		recognizer = pickle.loads(open(recognizerPath, "rb").read())
		le = pickle.loads(open(lePath, "rb").read())

		# initialize the video stream, then allow the camera sensor to warm up
		logger.info("starting video stream...")
		vs = VideoStream(src=0).start()
		# vs = VideoStream(usePiCamera=True).start()
		time.sleep(warmup_time)

		# start the FPS throughput estimator
		fps = FPS().start()

		# loop over frames from the video file stream
		people = ["unknown"]
		start_time = datetime.now()
		interval_time = datetime.now()
		record_name = False
		run_cam = True

		while run_cam:
			# grab the frame from the threaded video stream
			frame = vs.read()

			# resize the frame to have a width of 600 pixels (while
			# maintaining the aspect ratio), and then grab the image
			# dimensions
			frame = imutils.resize(frame, width=600)
			(h, w) = frame.shape[:2]

			# construct a blob from the image
			imageBlob = cv2.dnn.blobFromImage(
				cv2.resize(frame, (300, 300)), 1.0, (300, 300),
				(104.0, 177.0, 123.0), swapRB=False, crop=False)

			# apply OpenCV's deep learning-based face detector to localize
			# faces in the input image
			detector.setInput(imageBlob)
			detections = detector.forward()


			# loop over the detections
			for i in range(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated with
				# the prediction
				this_confidence = detections[0, 0, i, 2]

				# filter out weak detections
				if this_confidence > confidence:
					# compute the (x, y)-coordinates of the bounding box for
					# the face
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")

					# extract the face ROI
					face = frame[startY:endY, startX:endX]
					(fH, fW) = face.shape[:2]

					# ensure the face width and height are sufficiently large
					if fW < 20 or fH < 20:
						continue

					# construct a blob for the face ROI, then pass the blob
					# through our face embedding model to obtain the 128-d
					# quantification of the face
					# faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					# 	(96, 96), (0, 0, 0), swapRB=True, crop=False)
					# embedder.setInput(faceBlob)
					# vec = embedder.forward()

					# Use dlib encoding instead of cv embedder
					face_encodings = face_recognition.face_encodings(face)

					if face_encodings:

						name = "unknown"
						# perform classification to recognize the face
						preds = recognizer.predict_proba(face_encodings)[0]
						j = np.argmax(preds)
						proba = preds[j]
						name = le.classes_[j]

						if record_name:
							people.append(name)

						# draw the bounding box of the face along with the
						# associated probability
						text = "{}: {:.2f}%".format(name, proba * 100)
						y = startY - 10 if startY - 10 > 10 else startY + 10
						cv2.rectangle(frame, (startX, startY), (endX, endY),
							(0, 0, 255), 2)
						cv2.putText(frame, text, (startX, y),
							cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


			now_time = datetime.now()
			delta = now_time - start_time
			delta_interval = now_time - interval_time

			if delta_interval.microseconds > 200000:
				record_name = True
				interval_time = datetime.now()
			else:
				record_name = False

			# update the FPS counter
			fps.update()
			# show the output frame
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			if delta.seconds >= run_time or key == ord("q"):
				run_cam = False


		people_names = []

		people_dict = Counter(people)
		threshold = len(people)/len(people_dict)*0.8
		for k in people_dict.keys():
			if people_dict[k] >= threshold:
				people_names.append(k)

		if "unknown" in people_names:
			people_names.remove("unknown")
		logger.info("Person detected: {}".format(people_names))


		# stop the timer and display FPS information
		fps.stop()
		logger.info("elasped time: {:.2f}".format(fps.elapsed()))
		logger.info("approx. FPS: {:.2f}".format(fps.fps()))

		# do a bit of cleanup
		cv2.destroyAllWindows()
		vs.stop()

		return people_names
