import requests
from urllib.parse import quote_plus
import urllib

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s %(message)s')


def call(txt, lang="en-US", display="False"):

    logger.info(txt)
    payload = {'input': txt, 'lang': lang, 'display': display}
    params = urllib.parse.urlencode(payload, quote_via=urllib.parse.quote)
    r = requests.get('http://localhost:8081', params=params)

    logger.info(r.text)

    return r.text

def greet(persons, lang="en-US"):

    call(txt="Talk to Recognize Me", display="False")

    txt = "greet " + ",".join(persons)
    response = call(txt)

    return response

def recognize_me(txt):

    logger.info(txt)

    call(txt="Talk to Recognize Me", display="False")
    response = call(txt=txt, display="False")

    logger.info(response)

    s = response.split(";")
    response_dict = {x.split(":")[0]: x.split(":")[1] for x in s if ":" in x}

    return response_dict