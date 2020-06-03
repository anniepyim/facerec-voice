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
import subprocess
import os

from modules import ga_handler, recognize_video, porcupine_mic, face_trainer
from modules.mirror_handler import mirror_greet, mirror_spotify, mirror_spotify_status, mirror_youtube

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s %(message)s')

lastseen_limit = 10
lasttalk_limit = 60
lastheard_limit = 10
keepon = 60
lastseen_time = datetime.datetime.now() - datetime.timedelta(seconds=lastheard_limit)
lasttalk_time = datetime.datetime.now() - datetime.timedelta(seconds=lasttalk_limit)
lastmotion_time = datetime.datetime.now()

recognizer = recognize_video.RecognizerCam(run_time=3)
face_trainer = face_trainer.FaceTrainer(confidence=0.5)
wakeword_recognizer = porcupine_mic.PorcupineDemo()
wakeword_recognizer.daemon = True
wakeword_recognizer.start()
video_run = recognizer.status
spotify_status = None

video_run = False
persons = []

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
    yt_response = ""
    raw_url = ""

    if request.method == 'POST':

        logger.info(request.form)
        if "toggle_button" in request.form:

            video_resp = request.form['toggle_button']

            if video_resp == "Start":
                video_run = True
                video_run = recognizer.restart()
            else:
                video_run = False
                video_run = recognizer.stop()

        if "yt_url" in request.form and "submit_yt" in request.form:
            try:
                raw_url = request.form['yt_url']
                url = "https://www.youtube.com/embed/"+raw_url.split("youtu.be/")[1]+"?autoplay=1"
                play_youtube(url)
            except:
                yt_response = "Error in loading the URL!"

        if "stop_yt" in request.form:
            stop_youtube()
            logger.info("Youtube stopped")
            try:
                raw_url = request.form['yt_url']
            except:
                logger.info("no url")


    if video_run:
        video_button = "Stop"
    else:
        video_button = "Start"

    return render_template("index.html", button=video_button, yt_response=yt_response, raw_url=raw_url)


@app.route("/spotify")
def spotify():
    global spotify_changed

    spotify_changed = request.args.get('spotify_changed') == "True"

    return "Done"

@app.route("/screen")
def screen():

    screen_on = request.args.get('screen') == "on"

    if screen_on:
        os.system("vcgencmd display_power 1")
    else:
        os.system("vcgencmd display_power 0")

    return "Done"

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(recognizer.generate(),
                    mimetype = "multipart/x-mixed-replace; boundary=frame")


def wake_ga(lang="en-US"):
    global spotify_status,spotify_changed

    logger.info("Wakeword detected within timeframe")

    spotify_status = mirror_spotify_status()
    mirror_spotify("Pause Spotify")
    stop_youtube(True)

    response = ga_handler.call("ok google", lang)
    logger.info(response)

    if spotify_status and not spotify_changed:
        mirror_spotify("Play Spotify")


def train_ga():
    global spotify_status

    logger.info("Ask RecognizeMe to train for new person")

    spotify_status = mirror_spotify_status()
    mirror_spotify("Pause Spotify")
    stop_youtube(True)

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

    if spotify_status:
        mirror_spotify("Play Spotify")


def play_youtube(url):
    global spotify_status

    logger.info("Play youtube on MM")

    spotify_status = mirror_spotify_status()
    mirror_spotify("Pause Spotify")

    mirror_youtube("true", url)


def stop_youtube(all=False):
    global spotify_status

    mirror_youtube("false")
    if spotify_status and not all:
        mirror_spotify("Play Spotify")


def run():

    while True:
        global video_run, persons, lastmotion_time

        if video_run:
            # the function stops looping and continue if movement is detected
            recognizer.detect_motion()

            lastmotion_time = datetime.datetime.now()

            try:
                r = subprocess.run(['vcgencmd', 'display_power'], stdout=subprocess.PIPE)
                if 'display_power=0' in r.stdout.decode('utf-8'):
                    os.system("vcgencmd display_power 1")
            except:
                logger.info("No commands to turn on")

            # motion detcted, detect human faces
            persons = recognizer.recognize()

        time.sleep(0.5)

def trigger():

    lastseen_time = datetime.datetime.now() - datetime.timedelta(seconds=lastheard_limit)
    lasttalk_time = datetime.datetime.now() - datetime.timedelta(seconds=lasttalk_limit)

    while True:
        global persons, lastmotion_time

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
                mirror_greet(persons_str)
                # response = ga_handler.greet(persons)
                # if response:
                #     logger.info(response)

                lasttalk_time = datetime.datetime.now()

        from_lastmotion = datetime.datetime.now() - lastmotion_time

        try:
            if from_lastmotion.seconds >= keepon:
                os.system("vcgencmd display_power 0")
        except:
            logger.info("No commands to turn off")

        persons = []
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

    # start a thread that will perform motion detection
    t2 = threading.Thread(target=trigger)
    t2.daemon = True
    t2.start()

    stop_youtube()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)
