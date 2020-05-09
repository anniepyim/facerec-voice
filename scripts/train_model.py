# USAGE
# python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle
# python train_model.py

# import the necessary packages
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", default="resources/face_detection_output/embeddings.pickle",
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", default="resources/face_detection_output/recognizer.pickle",
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", default="resources/face_detection_output/le.pickle",
	help="path to output label encoder")
args = vars(ap.parse_args())

# load the face embeddings
logger.info("loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

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
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()