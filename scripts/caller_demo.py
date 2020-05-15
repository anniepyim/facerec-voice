import requests
import sys
from urllib.parse import quote_plus
import urllib

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

txt = None

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
