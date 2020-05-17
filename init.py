import time
import datetime

from modules import ga_handler, recognize_video, infrared, porcupine_mic
from modules.train_new_person import build_face_dataset, extract_embeddings, train_model

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s %(message)s')

lastseen_limit = 10
lasttalk_limit = 60
lastheard_limit = 10

infrared_detector = infrared.InfraredDetector()
recognizer = recognize_video.RecognizerCam()
wakeword_recognizer = porcupine_mic.PorcupineDemo()
lastseen_time = datetime.datetime.now() - datetime.timedelta(seconds=lastheard_limit)
lasttalk_time = datetime.datetime.now() - datetime.timedelta(seconds=lasttalk_limit)
wake_ga = False

wakeword_recognizer.daemon = True
wakeword_recognizer.start()


def wake_ga(lang="en-US"):

    logger.info("Wakeword detected within timeframe")
    logger.info("See {} again. Waking up Assistant".format(",".join(persons)))
    response = ga_handler.call("ok google", lang)
    logger.info(response)

def train_ga():

    logger.info("Ask RecognizeMe to train for new person")
    res_dict = ga_handler.recognize_me("meet a new friend")

    logger.info(res_dict)

    if ("new_person" in res_dict and "response" in res_dict):
        if res_dict["response"]:
            new_person = res_dict["new_person"]
            logger.info("Take pictures for {}".format(new_person))
            build_face_dataset(new_person)
            logger.info("Extract embeddings for {}".format(new_person))
            extract_embeddings()
            logger.info("Train model to add {}".format(new_person))
            train_model()
    elif "skip_fallback" not in res_dict:
        ga_handler.call("stop")

while True:

    # Run detector forever
    detected = infrared_detector.run()

    # response when detected movement
    if detected:

        # detect trained faces
        persons = recognizer.run()

        # response when detected known person
        if len(persons) > 0:
            lastseen_time = datetime.datetime.now()

    # the time from when the system last saw a known person
    from_lastseen = datetime.datetime.now() - lastseen_time

    # the time from when the system last heard wakeword
    wakeword_lastheard = wakeword_recognizer.detected_time
    from_lastheard = datetime.datetime.now() - wakeword_lastheard
    from_lasttalk = datetime.datetime.now() - lasttalk_time

    # give appropriate response if know person is detected
    if from_lastseen.seconds < lastseen_limit:

        # interact with user if wakeword is heard
        if from_lastheard.seconds < lastheard_limit:
            if wakeword_recognizer.detected_keyword == "porcupine":
                wake_ga()
            if wakeword_recognizer.detected_keyword == "grasshopper":
                wake_ga("de-DE")
            if wakeword_recognizer.detected_keyword == "bumblebee":
                train_ga()

            lasttalk_time = datetime.datetime.now()
            # reset detected time so it won't be called again
            wakeword_recognizer.reset_detected()

        # else greet the person if beyond the last seen time
        elif from_lasttalk.seconds >= lasttalk_limit:
            logger.info("See {} for the first time".format(",".join(persons)))
            response = ga_handler.greet(persons)
            if response:
                logger.info(response)

            lasttalk_time = datetime.datetime.now()

    time.sleep(0.1)

