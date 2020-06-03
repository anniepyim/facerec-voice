import urllib
import requests
import json
import spacy
nlp = spacy.load("en_core_web_sm")

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s %(message)s')

port = '8123'
port_server = '8082'

def keep_token(t):
    return (t.tag_ == "NN" or t.tag_ == "NNP" or t.tag_ == "NNS" or t.tag_ == "NNPS") \
           and not (t.text == "spotify" or t.text == "Spotify" or t.text == 'music')

def mirror_call(notification, reply):
    payload = {'notification': notification, 'reply': reply}
    params = urllib.parse.urlencode(payload, quote_via=urllib.parse.quote)
    r = requests.get('http://localhost:'+port+'/ga?', params=params)

    logger.info(r.text)

def mirror_greet(persons):

    payload = {'type': 'INFO', 'message': persons, 'silent': 'true'}
    params = urllib.parse.urlencode(payload, quote_via=urllib.parse.quote)
    r = requests.get('http://localhost:'+port+'/greetings?', params=params)

    logger.info(r.text)

def mirror_youtube(load, url=""):

    payload = {'load': load, 'embed_url': url}
    params = urllib.parse.urlencode(payload, quote_via=urllib.parse.quote)
    r = requests.get('http://localhost:'+port+'/youtube?', params=params)

    logger.info(r.text)

def mirror_spotify(user_response, send_status=False):

    try:
        s = user_response[0].lower() + user_response[1:]
    except:
        return False

    doc = nlp(s)
    query = " ".join([t.text for t in doc if keep_token(t)])
    type = "playlist"

    if ("spotify" in s or "Spotify" in s):
        if ("stop" in s) or ("pause" in s):
            payload = {'notification': 'PAUSE'}
        elif len(query) > 0:
            payload = {'notification': 'SEARCH_AND_PLAY', 'q': query, 'type': type, 'random': True,
                       'autoplay': True}
        else:
            payload = {'notification': 'PLAY'}
        r = requests.get('http://localhost:' + port_server + '/spotify?spotify_changed=' + str(send_status))
        params = urllib.parse.urlencode(payload, quote_via=urllib.parse.quote)
        r = requests.get('http://localhost:'+port+'/spotify?', params=params)
        logger.info(r.text)
        return True
    else:
        return False

def mirror_spotify_status():
    r = requests.get('http://localhost:8123/spotify?', params='notification=GET_STATUS')
    status = json.loads(r.text)['is_playing'] == "true"
    return status
