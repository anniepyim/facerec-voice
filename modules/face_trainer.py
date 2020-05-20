#import the necessary packages
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np
import pickle
import cv2
import os
import imutils
from imutils import paths
import face_recognition

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s %(message)s')

cascadePath = os.path.join(os.path.dirname(__file__),
                           "../resources/haarcascade/haarcascade_frontalface_default.xml")
protoPath = os.path.join(os.path.dirname(__file__),
                         "../resources/face_detection_model/deploy.prototxt")
modelPath = os.path.join(os.path.dirname(__file__),
                         "../resources/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")
dataPath = os.path.join(os.path.dirname(__file__),
                        "../dataset/face_detection")
embeddingPath = os.path.join(os.path.dirname(__file__),
                             "../resources/face_detection_output/embeddings.pickle")
# embedderPath = os.path.join(os.path.dirname(__file__),
# 						  "../resources/face_embedding_model/openface_nn4.small2.v1.t7")
# load our serialized face embedding model from disk
# logger.info("loading face recognizer...")
# embedder = cv2.dnn.readNetFromTorch(embedderPath)
# embedder.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

recognizerPath = os.path.join(os.path.dirname(__file__),
                              "../resources/face_detection_output/recognizer.pickle")
lePath = os.path.join(os.path.dirname(__file__),
                      "../resources/face_detection_output/le.pickle")

class FaceTrainer():

    def __init__(self, confidence=0.5):

        self.confidence = confidence

        # # load OpenCV's Haar cascade for face detection from disk
        # self.detector = cv2.CascadeClassifier(cascadePath)

        # load our serialized face detector from disk
        logger.info("loading face detector...")
        self.detector_extract = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        # detector.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)


    def extract_embeddings(self):

        # grab the paths to the input images in our dataset
        logger.info("quantifying faces...")

        imagePaths = list(paths.list_images(dataPath))

        # initialize our lists of extracted facial embeddings and
        # corresponding people names
        knownEmbeddings = []
        knownNames = []

        # initialize the total number of faces processed
        total = 0

        # loop over the image paths
        for (i, imagePath) in enumerate(imagePaths):
            # extract the person name from the image path
            logger.info("processing image {}/{}".format(i + 1,
                len(imagePaths)))
            name = imagePath.split(os.path.sep)[-2]

            # load the image, resize it to have a width of 600 pixels (while
            # maintaining the aspect ratio), and then grab the image
            # dimensions
            image = cv2.imread(imagePath)
            image = imutils.resize(image, width=600)
            (h, w) = image.shape[:2]

            # construct a blob from the image
            imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

            # apply OpenCV's deep learning-based face detector to localize
            # faces in the input image
            self.detector_extract.setInput(imageBlob)
            detections = self.detector_extract.forward()

            # ensure at least one face was found
            if len(detections) > 0:
                # we're making the assumption that each image has only ONE
                # face, so find the bounding box with the largest probability
                i = np.argmax(detections[0, 0, :, 2])
                this_confidence = detections[0, 0, i, 2]

                # ensure that the detection with the largest probability also
                # means our minimum probability test (thus helping filter out
                # weak detections)
                if this_confidence > self.confidence:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # extract the face ROI and grab the ROI dimensions
                    face = image[startY:endY, startX:endX]
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
                        # add the name of the person + corresponding face
                        # embedding to their respective lists
                        knownNames.append(name)
                        knownEmbeddings.append(face_encodings[0].flatten())
                        total += 1

        # dump the facial embeddings + names to disk
        logger.info("serializing {} encodings...".format(total))
        data = {"embeddings": knownEmbeddings, "names": knownNames}
        f = open(embeddingPath, "wb")
        f.write(pickle.dumps(data))
        f.close()

    def train_model(self):
        # load the face embeddings
        logger.info("loading face embeddings...")
        data = pickle.loads(open(embeddingPath, "rb").read())

        # encode the labels
        logger.info("encoding labels...")
        le = LabelEncoder()
        labels = le.fit_transform(data["names"])

        # train the model used to accept the 128-d embeddings of the face and
        # then produce the actual face recognition
        logger.info("training model...")
        params = {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "gamma": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]}
        recognizer = GridSearchCV(SVC(kernel="rbf", gamma="auto",
            probability=True), params, cv=3, n_jobs=-1)
        recognizer.fit(data["embeddings"], labels)
        logger.info("best hyperparameters: {}".format(recognizer.best_params_))

        # write the actual face recognition model to disk
        f = open(recognizerPath, "wb")
        f.write(pickle.dumps(recognizer))
        f.close()

        # write the label encoder to disk
        f = open(lePath, "wb")
        f.write(pickle.dumps(le))
        f.close()
