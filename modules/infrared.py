import keyboard
import datetime
import os

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s %(message)s')

class InfraredDetector():

    def __init__(self):
        self.have_detected = False
        self.first = True
        self.trigger_time = datetime.datetime.now()

    def run(self):

        if keyboard.is_pressed('q') and (not self.have_detected):  # if key 'q' is pressed
            logger.info('You Pressed A Key!')
            self.have_detected = True
            return True

        elif (not keyboard.is_pressed('q')) and self.have_detected:
            self.have_detected = False
            return False


