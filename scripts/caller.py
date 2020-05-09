import requests
import sys
from urllib.parse import quote_plus
import urllib
import time

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

#import pyttsx3

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

txt = None

# engine = pyttsx3.init()
# voices = engine.getProperty('voices')
# rate = engine.getProperty('rate')
#
# # for voice in voices:
# #     print(voice.id)
# #     print(voice)
# engine.setProperty('voice', "com.apple.speech.synthesis.voice.samantha")
#
# engine.say('Ready? ; Three')
#
# engine.runAndWait()

# engine.say('Three')
#
# #engine.runAndWait()
#
# engine.say('Two')
# #engine.runAndWait()
#
# engine.say('One')
# #engine.runAndWait()

# engine.say('Smile!')
# engine.runAndWait()


# engine.say("I will speak this text", "something else")
# engine.runAndWait()


# engine.say("Ready")
# engine.runAndWait()
#
# time.sleep(1)
# engine.say("3")
# engine.runAndWait()
#
# time.sleep(1)
# engine.say("2")
# engine.runAndWait()
# time.sleep(1)
#
# engine.say("1")
# engine.runAndWait()
# time.sleep(1)
#
# engine.say("Smile")
# engine.runAndWait()
# time.sleep(1)


#
while txt != 'quit':


    txt = input("Type something to test this out: ")
    lang = input("Language?: ")

    if lang == '':
        lang = 'en_US'

    payload = {'input': txt, 'lang': lang}
    #headers = {'Content-Type': 'application/json;charset=UTF-8'}
    params = urllib.parse.urlencode(payload, quote_via=urllib.parse.quote)
    r = requests.get('http://localhost:8081', params=params)

    logger.info(r.text)