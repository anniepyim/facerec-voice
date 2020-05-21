# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from pathlib import Path
import numpy as np
import imutils
import pickle
import time
import cv2
import os
import face_recognition
from collections import Counter
from datetime import datetime

import threading
lock = threading.Lock()
outputFrame = None

from . import ga_handler
from . import singlemotiondetector

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

# grab global references to the video stream, output frame, and
# lock variables

class RecognizerCam():

    def __init__(self, confidence=0.6, run_time=5, warmup_time=1.0):

        self.confidence = confidence
        self.run_time = run_time

        # load our serialized face detector from disk
        logger.info("Loading detector and recognizer")
        self.detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        # self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

        # load our serialized face embedding model from disk
        # preferable target to MYRIAD
        # self.embedder = cv2.dnn.readNetFromTorch(embedderPath)
        # self.embedder.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

        # load the actual face recognition model along with the label encoder
        self.recognizer = pickle.loads(open(recognizerPath, "rb").read())
        self.le = pickle.loads(open(lePath, "rb").read())

        # initialize the video stream, then allow the camera sensor to warm up
        self.warmup_time = warmup_time
        self.status = True
        logger.info("starting video stream...")
        self.vs = VideoStream(src=0).start()
        # vs = VideoStream(usePiCamera=True).start()
        time.sleep(self.warmup_time)

    def restart(self):
        self.vs = VideoStream(src=0).start()
        time.sleep(self.warmup_time)
        self.status = True

        return self.status

    def stop(self):
        self.vs.stop()
        time.sleep(2.0)
        self.vs.stream.release()
        time.sleep(2.0)
        self.status = False

        return self.status

    def detect_motion(self, frameCount=10):
        # grab global references to the video stream, output frame, and
        # lock variables
        global outputFrame, lock

        # initialize the motion detector and the total number of frames
        # read thus far
        md = singlemotiondetector.SingleMotionDetector(accumWeight=0.1)
        total = 0

        # loop over frames from the video stream
        while True and self.status:
            motion = None

            # read the next frame from the video stream, resize it,
            # convert the frame to grayscale, and blur it
            frame = self.vs.read()
            frame = imutils.resize(frame, width=600)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            # grab the current timestamp and draw it on the frame
            timestamp = datetime.now()
            cv2.putText(frame, timestamp.strftime(
                "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            # if the total number of frames has reached a sufficient
            # number to construct a reasonable background model, then
            # continue to process the frame
            if total > frameCount:
                # detect motion in the image
                motion = md.detect(gray)

                # cehck to see if motion was found in the frame
                if motion is not None:
                    # unpack the tuple and draw the box surrounding the
                    # "motion area" on the output frame
                    (thresh, (minX, minY, maxX, maxY)) = motion
                    cv2.rectangle(frame, (minX, minY), (maxX, maxY),
                                  (0, 0, 255), 2)

            # update the background model and increment the total number
            # of frames read thus far
            md.update(gray)
            total += 1

            time.sleep(0.25)
            # show the output frame
            # cv2.imshow("Frame", frame)
            # key = cv2.waitKey(1) & 0xFF

            # acquire the lock, set the output frame, and release the
            # lock
            with lock:
                outputFrame = frame.copy()

            if motion is not None:
                break

    def recognize(self):
        # grab global references to the video stream, output frame, and
        # lock variables
        global outputFrame, lock

        # start the FPS throughput estimator
        fps = FPS().start()

        # loop over frames from the video file stream
        people = ["unknown"]
        start_time = datetime.now()
        interval_time = datetime.now()
        record_name = False
        run_cam = True

        while run_cam and self.status:
            # grab the frame from the threaded video stream
            frame = self.vs.read()

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
            self.detector.setInput(imageBlob)
            detections = self.detector.forward()


            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                this_confidence = detections[0, 0, i, 2]

                # filter out weak detections
                if this_confidence > self.confidence:
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
                        preds = self.recognizer.predict_proba(face_encodings)[0]
                        j = np.argmax(preds)
                        proba = preds[j]
                        name = self.le.classes_[j]

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
            cv2.putText(frame, "Recognizing...", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

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
            time.sleep(0.25)
            # show the output frame
            # cv2.imshow("Frame", frame)
            # key = cv2.waitKey(1) & 0xFF

            with lock:
                outputFrame = frame.copy()

            # if the `q` key was pressed, break from the loop
            # if delta.seconds >= self.run_time or key == ord("q"):
            if delta.seconds >= self.run_time:
                run_cam = False


        people_names = []

        people_dict = Counter(people)
        threshold = len(people)/len(people_dict)*0.8
        for k in people_dict.keys():
            if people_dict[k] >= threshold:
                people_names.append(k)

        if "unknown" in people_names:
            people_names.remove("unknown")

        logger.info("raw dict: {}".format(people_dict))
        logger.info("Person detected: {}".format(people_names))


        # stop the timer and display FPS information
        fps.stop()
        # logger.info("elasped time: {:.2f}".format(fps.elapsed()))
        # logger.info("approx. FPS: {:.2f}".format(fps.fps()))

        # do a bit of cleanup
        cv2.destroyAllWindows()
        #vs.stop()

        return people_names

    def build_face_dataset(self, person):
        # grab global references to the video stream, output frame, and
        # lock variables
        global outputFrame, lock

        # initialize the video stream, allow the camera sensor to warm up,
        # and initialize the total number of example faces written to disk
        # thus far

        outputPath = os.path.join(os.path.dirname(__file__),
                                  "../dataset/face_detection", person)
        Path(outputPath).mkdir(parents=True, exist_ok=True)
        # logger.info("starting video stream...")
        # vs = VideoStream(src=0).start()
        # vs = VideoStream(usePiCamera=True).start()
        # time.sleep(1.0)
        total = 0
        res = ga_handler.call(txt="Talk to Recognize Me", display="False")
        # loop over the frames from the video stream
        while res:
            # grab the frame from the threaded video stream, clone it, (just
            # in case we want to write it to disk), and then resize the frame
            # so we can apply face detection faster

            if total == 0:
                txt = "Take the first picture"
                res_dict = ga_handler.call(txt=txt, display="False")
                logger.info(res_dict)
            elif total == 4:
                txt = "Take the last picture"
                res_dict = ga_handler.call(txt=txt, display="False")
                logger.info(res_dict)
            else:
                txt = "Take more pictures"
                res_dict = ga_handler.call(txt=txt, display="False")
                logger.info(res_dict)

            frame = self.vs.read()
            # orig = frame.copy()
            # frame = imutils.resize(frame, width=400)
            #
            # # detect faces in the grayscale frame
            # rects = detector.detectMultiScale(
            # 	cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
            # 	minNeighbors=5, minSize=(30, 30))
            #
            # # loop over the face detections and draw them on the frame
            # for (x, y, w, h) in rects:
            # 	cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #
            # # show the output frame
            # cv2.imshow("Frame", frame)
            # key = cv2.waitKey(1) & 0xFF

            # if the `k` key was pressed, write the *original* frame to disk
            # so we can later process it and use it for face recognition

            p = os.path.sep.join([outputPath, "{}.png".format(
                str(total).zfill(5))])
            cv2.imwrite(p, frame)
            total += 1

            # acquire the lock, set the output frame, and release the
            # lock
            with lock:
                outputFrame = frame.copy()

            # if the `q` key was pressed, break from the loop
            # elif key == ord("q"):
            if total >= 5:
                print("Done!")
                break

        # do a bit of cleanup
        logger.info("{} face images stored".format(total))
        logger.info("cleaning up...")
        cv2.destroyAllWindows()
        # vs.stop()

    def reset_model(self):
        logger.info("Reloading model")
        self.recognizer = pickle.loads(open(recognizerPath, "rb").read())
        self.le = pickle.loads(open(lePath, "rb").read())

    def generate(self):
        # grab global references to the output frame and lock variables
        global outputFrame, lock

        # loop over frames from the output stream
        while True and self.status:
            # wait until the lock is acquired
            with lock:
                # check if the output frame is available, otherwise skip
                # the iteration of the loop
                if outputFrame is None:
                    continue

                # encode the frame in JPEG format
                (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

                # ensure the frame was successfully encoded
                if not flag:
                    continue

            # yield the output frame in the byte format
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                   bytearray(encodedImage) + b'\r\n')

            time.sleep(0.25)
