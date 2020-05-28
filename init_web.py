# USAGE
# python textandtalk_http.py --port 8081
# python init_webstreaming.py --ip 0.0.0.0 --port 8082
# ngrok http -auth="username:password" 8082

# import the necessary packages
from flask import Response, Flask, render_template
from flask import abort, request
import threading
import argparse
#import keyboard
import datetime
import time
import requests
import urllib

from modules import ga_handler, recognize_video, porcupine_mic, face_trainer

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s %(message)s')

lastseen_limit = 10
lasttalk_limit = 60
lastheard_limit = 10
lastseen_time = datetime.datetime.now() - datetime.timedelta(seconds=lastheard_limit)
lasttalk_time = datetime.datetime.now() - datetime.timedelta(seconds=lasttalk_limit)

recognizer = recognize_video.RecognizerCam(run_time=5)
face_trainer = face_trainer.FaceTrainer(confidence=0.5)
wakeword_recognizer = porcupine_mic.PorcupineDemo()
wakeword_recognizer.daemon = True
wakeword_recognizer.start()
video_run = recognizer.status

ALLOWED_IPS = ['127.0.0.1']

# initialize a flask object
app = Flask(__name__)

@app.before_request
def limit_remote_addr():
    if request.remote_addr not in ALLOWED_IPS:
        abort(403)  # Forbidden


@app.route("/", methods=['GET', 'POST'])
def index():
    global video_run
    # return the rendered template

    video_run = recognizer.status

    if request.method == 'POST':
        video_resp = request.form['toggle_button']

        if video_resp == "Start":
            video_run = recognizer.restart()
        else:
            video_run = recognizer.stop()

    if video_run:
        video_button = "Stop"
    else:
        video_button = "Start"

    return render_template("index.html", button=video_button)


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(recognizer.generate(),
                    mimetype = "multipart/x-mixed-replace; boundary=frame")


def wake_ga(lang="en-US"):

    logger.info("Wakeword detected within timeframe")
    response = ga_handler.call("ok google", lang)
    logger.info(response)


def train_ga():

    logger.info("Ask RecognizeMe to train for new person")
    res_dict = ga_handler.recognize_me("meet a new friend")

    logger.info(res_dict)

    if "new_person" in res_dict and "response" in res_dict:
        if res_dict["response"]:
            new_person = res_dict["new_person"]
            logger.info("Take pictures for {}".format(new_person))
            recognizer.build_face_dataset(new_person)
            logger.info("Extract embeddings for {}".format(new_person))
            face_trainer.extract_embeddings()
            logger.info("Train model to add {}".format(new_person))
            face_trainer.train_model()

            # reload model after training
            recognizer.reset_model()

    elif "skip_fallback" not in res_dict:
        ga_handler.call("stop")

def call_mirror_greet(persons):

    payload = {'type': 'INFO', 'message': persons, 'silent': 'true'}
    params = urllib.parse.urlencode(payload, quote_via=urllib.parse.quote)
    r = requests.get('http://localhost:8080/greetings?', params=params)

    logger.info(r.text)

def run():

    lastseen_time = datetime.datetime.now() - datetime.timedelta(seconds=lastheard_limit)
    lasttalk_time = datetime.datetime.now() - datetime.timedelta(seconds=lasttalk_limit)

    while True:
        global video_run

        if video_run:
            # the function stops looping and continue if movement is detected
            recognizer.detect_motion()

            # motion detcted, detect human faces
            persons = recognizer.recognize()

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

                    if len(persons) > 1:
                        persons_str = '{} and {}'.format(', '.join(persons[:-1]), persons[-1])
                    else:
                        persons_str = persons[0]

                    logger.info("See {} for the first time".format(persons_str))
                    call_mirror_greet(persons_str)
                    # response = ga_handler.greet(persons)
                    # if response:
                    #     logger.info(response)

                    lasttalk_time = datetime.datetime.now()

        time.sleep(0.5)


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, default='0.0.0.0',
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, default=8082,
        help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
        help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    # start a thread that will perform motion detection
    t = threading.Thread(target=run)
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)
