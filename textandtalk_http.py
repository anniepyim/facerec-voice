# USAGE
# python textandtalk_http.py --port 8081

import argparse
import concurrent.futures
import json
import os
import os.path
import pathlib2 as pathlib
import sys
import time
import uuid

import click
import grpc
import google.auth.transport.grpc
import google.auth.transport.requests
import google.oauth2.credentials

from google.assistant.embedded.v1alpha2 import (
    embedded_assistant_pb2,
    embedded_assistant_pb2_grpc
)
from tenacity import retry, stop_after_attempt, retry_if_exception

try:
    from googlesamples.assistant.grpc import (
        assistant_helpers,
        audio_helpers,
        device_helpers
    )
except (SystemError, ImportError):
    import assistant_helpers
    import audio_helpers
    import device_helpers

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from urllib.parse import unquote
import urllib
import requests

from modules import browser_helpers

import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class SampleAssistant(object):
    """Sample Assistant that supports conversations and device actions.

    Args:
      device_model_id: identifier of the device model.
      device_id: identifier of the registered device instance.
      conversation_stream(ConversationStream): audio stream
        for recording query and playing back assistant answer.
      channel: authorized gRPC channel for connection to the
        Google Assistant API.
      deadline_sec: gRPC deadline in seconds for Google Assistant API call.
      device_handler: callback for device actions.
    """

    def __init__(self, language_code, device_model_id, device_id,
                 conversation_stream, display,
                 channel, deadline_sec, device_handler):
        self.language_code = language_code
        self.device_model_id = device_model_id
        self.device_id = device_id
        self.conversation_stream = conversation_stream
        self.display = display

        self.END_OF_UTTERANCE = embedded_assistant_pb2.AssistResponse.END_OF_UTTERANCE
        self.DIALOG_FOLLOW_ON = embedded_assistant_pb2.DialogStateOut.DIALOG_FOLLOW_ON
        self.CLOSE_MICROPHONE = embedded_assistant_pb2.DialogStateOut.CLOSE_MICROPHONE
        self.PLAYING = embedded_assistant_pb2.ScreenOutConfig.PLAYING

        # Opaque blob provided in AssistResponse that,
        # when provided in a follow-up AssistRequest,
        # gives the Assistant a context marker within the current state
        # of the multi-Assist()-RPC "conversation".
        # This value, along with MicrophoneMode, supports a more natural
        # "conversation" with the Assistant.
        self.conversation_state = None
        # Force reset of first conversation.
        self.is_new_conversation = True

        # Create Google Assistant API gRPC client.
        self.assistant = embedded_assistant_pb2_grpc.EmbeddedAssistantStub(
            channel
        )
        self.deadline = deadline_sec

        self.device_handler = device_handler

    def __enter__(self):
        return self

    def __exit__(self, etype, e, traceback):
        if e:
            return False
        self.conversation_stream.close()

    def is_grpc_error_unavailable(e):
        is_grpc_error = isinstance(e, grpc.RpcError)
        if is_grpc_error and (e.code() == grpc.StatusCode.UNAVAILABLE):
            logging.error('grpc unavailable error: %s', e)
            return True
        return False

    @retry(reraise=True, stop=stop_after_attempt(3),
           retry=retry_if_exception(is_grpc_error_unavailable))
    def assist(self, text_query=None, language_code='en-US', display=None):
        """Send a voice request to the Assistant and playback the response.

        Returns: True if conversation should continue.
        """
        text_response = None
        continue_conversation = False
        give_audio = True
        device_actions_futures = []
        user_response = None
        self.language_code = language_code
        if display:
            self.display = display == "True"

        def iter_log_assist_requests():
            for c in self.gen_assist_requests(text_query):
                assistant_helpers.log_assist_request_without_audio(c)
                yield c
            logging.debug('Reached end of AssistRequest iteration.')

        if text_query is None:
            self.conversation_stream.start_recording()
            logger.info('Recording audio request.')

        # This generator yields AssistResponse proto messages
        # received from the gRPC Google Assistant API.
        for resp in self.assistant.Assist(iter_log_assist_requests(),
                                          self.deadline):
            assistant_helpers.log_assist_response_without_audio(resp)

            if text_query is None:
                if resp.event_type == self.END_OF_UTTERANCE:
                    logger.info('End of audio request detected.')
                    logger.info('Stopping recording.')
                    call_mirror("user_reply", user_response)
                    self.conversation_stream.stop_recording()
                if resp.speech_results:
                    user_response = ' '.join(r.transcript for r in resp.speech_results)
                    logger.info('Transcript of user request: "%s".', user_response)
            if resp.dialog_state_out.supplemental_display_text:
                text_response = resp.dialog_state_out.supplemental_display_text

            if text_response is not None:

                s = text_response.split(";")
                res_dct = {x.split(":")[0]: x.split(":")[1] for x in s if ":" in x}

                if "RecognizeMe_talk" in res_dct:
                    if res_dct["RecognizeMe_talk"] == "False":
                        give_audio = False
                    else:
                        give_audio = True

            if len(resp.audio_out.audio_data) > 0 and give_audio:
                if not self.conversation_stream.playing:
                    self.conversation_stream.stop_recording()
                    self.conversation_stream.start_playback()
                    logger.info('Playing assistant response.')
                self.conversation_stream.write(resp.audio_out.audio_data)
            if resp.dialog_state_out.conversation_state:
                conversation_state = resp.dialog_state_out.conversation_state
                logging.debug('Updating conversation state.')
                self.conversation_state = conversation_state
            if resp.dialog_state_out.volume_percentage != 0:
                volume_percentage = resp.dialog_state_out.volume_percentage
                logger.info('Setting volume to %s%%', volume_percentage)
                self.conversation_stream.volume_percentage = volume_percentage
            if resp.dialog_state_out.microphone_mode == self.DIALOG_FOLLOW_ON:
                continue_conversation = True
                logger.info('Expecting follow-on query from user.')
            elif resp.dialog_state_out.microphone_mode == self.CLOSE_MICROPHONE:
                continue_conversation = False
            # if resp.device_action.device_request_json:
            #     device_request = json.loads(
            #         resp.device_action.device_request_json
            #     )
            #     fs = self.device_handler(device_request)
            #     if fs:
            #         device_actions_futures.extend(fs)
            if resp.screen_out.data:
                system_browser = browser_helpers.system_browser
                system_browser.display(resp.screen_out.data)

        if len(device_actions_futures):
            logger.info('Waiting for device executions to complete.')
            concurrent.futures.wait(device_actions_futures)

        logger.info('Finished playing assistant response.')
        self.conversation_stream.stop_playback()

        if text_response:
            logger.info(text_response)

        return continue_conversation, text_response

    def gen_assist_requests(self, text_query=None):
        """Yields: AssistRequest messages to send to the API."""

        config = embedded_assistant_pb2.AssistConfig(
            audio_in_config=embedded_assistant_pb2.AudioInConfig(
                encoding='LINEAR16',
                sample_rate_hertz=self.conversation_stream.sample_rate,
            ),
            audio_out_config=embedded_assistant_pb2.AudioOutConfig(
                encoding='LINEAR16',
                sample_rate_hertz=self.conversation_stream.sample_rate,
                volume_percentage=self.conversation_stream.volume_percentage,
            ),
            dialog_state_in=embedded_assistant_pb2.DialogStateIn(
                language_code=self.language_code,
                conversation_state=self.conversation_state,
                is_new_conversation=self.is_new_conversation,
            ),
            device_config=embedded_assistant_pb2.DeviceConfig(
                device_id=self.device_id,
                device_model_id=self.device_model_id,
            ),
            text_query=text_query
        )
        if self.display:
            config.screen_out_config.screen_mode = self.PLAYING
        # Continue current conversation with later requests.
        self.is_new_conversation = False
        # The first AssistRequest must contain the AssistConfig
        # and no audio data.
        yield embedded_assistant_pb2.AssistRequest(config=config)
        if text_query is None:
            for data in self.conversation_stream:
                # Subsequent requests need audio data, but not config.
                yield embedded_assistant_pb2.AssistRequest(audio_in=data)

# HTTPRequestHandler class
class testHTTPServer_RequestHandler(BaseHTTPRequestHandler):

    # GET
    def do_GET(self):

        # Send response status code
        self.send_response(200)

        # Send headers
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        query = urlparse(self.path).query
        query_components = dict(qc.split("=") for qc in query.split("&"))

        # Retrieve input text
        input_text = unquote(query_components["input"])
        lang = unquote(query_components["lang"])
        display = unquote(query_components["display"])

        responses = continue_audio_handler(input_text, lang, display)

        if responses != [None]:
            responses_str = (';').join(responses)
            self.wfile.write(bytes(responses_str, "utf8"))

        return

def call_mirror(notification, reply):
    payload = {'notification': notification, 'reply': reply}
    params = urllib.parse.urlencode(payload, quote_via=urllib.parse.quote)
    r = requests.get('http://localhost:8080/ga?', params=params)

    logger.info(r.text)

def recognizeme_audio(s):

    audio = True

    if s and len(s.split()) >= 1:
        s = s.split(";")
        res_dct = {x.split(":")[0]: x.split(":")[1] for x in s if ":" in x}

        if "RecognizeMe_continue_audio" in res_dct :
            if res_dct["RecognizeMe_continue_audio"] == "False":
                audio = False

    return audio

def continue_audio_handler(input_text, lang="en_US", display=None):

    response_text_strs = []

    continue_conversation, response_text = assistant.assist(text_query=input_text, language_code=lang, display=display)

    response_text_strs.append(response_text)

    if recognizeme_audio(response_text) and continue_conversation:

        continue_audio = True
    else:
        continue_audio = False

    while continue_audio:

        if continue_conversation:
            call_mirror("conv_on", "")
            # os.system("aplay resources/soundwav/start.wav")

        continue_conversation, response_text = assistant.assist()
        continue_audio = continue_conversation

        if response_text:
            response_text_strs.append(response_text)

    call_mirror("conv_off", "")
    # os.system("aplay resources/soundwav/stop.wav")

    return response_text_strs

def run(port):

    api_endpoint = 'embeddedassistant.googleapis.com'
    project_id = "assistantdevice-1f1e8"
    device_model_id = "assistantdevice-1f1e8-ga1-v3slut"
    device_id = 'assistantdevice-1f1e8'
    credentials = os.path.join(click.get_app_dir('google-oauthlib-tool'),
                               'credentials.json')
    device_config = os.path.join(click.get_app_dir('googlesamples-assistant'),
                                 'device_config.json')
    lang = "en-US"
    display = False
    verbose = False
    input_audio_file = None
    output_audio_file = None
    audio_sample_rate = audio_helpers.DEFAULT_AUDIO_SAMPLE_RATE
    audio_sample_width = audio_helpers.DEFAULT_AUDIO_SAMPLE_WIDTH
    audio_iter_size = audio_helpers.DEFAULT_AUDIO_ITER_SIZE
    audio_block_size = audio_helpers.DEFAULT_AUDIO_DEVICE_BLOCK_SIZE
    audio_flush_size = audio_helpers.DEFAULT_AUDIO_DEVICE_FLUSH_SIZE
    grpc_deadline = 60 * 3 + 5
    once = False

    global assistant

    # Load OAuth 2.0 credentials.
    try:
        with open(credentials, 'r') as f:
            credentials = google.oauth2.credentials.Credentials(token=None,
                                                                **json.load(f))
            http_request = google.auth.transport.requests.Request()
            credentials.refresh(http_request)
    except Exception as e:
        logger.error('Error loading credentials: %s', e)
        logger.error('Run google-oauthlib-tool to initialize '
                      'new OAuth 2.0 credentials.')
        sys.exit(-1)

    # Create an authorized gRPC channel.
    grpc_channel = google.auth.transport.grpc.secure_authorized_channel(
        credentials, http_request, api_endpoint)
    logger.info('Connecting to %s', api_endpoint)

    # Configure audio source and sink.
    audio_device = None
    if input_audio_file:
        audio_source = audio_helpers.WaveSource(
            open(input_audio_file, 'rb'),
            sample_rate=audio_sample_rate,
            sample_width=audio_sample_width
        )
    else:
        audio_source = audio_device = (
            audio_device or audio_helpers.SoundDeviceStream(
                sample_rate=audio_sample_rate,
                sample_width=audio_sample_width,
                block_size=audio_block_size,
                flush_size=audio_flush_size
            )
        )
    if output_audio_file:
        audio_sink = audio_helpers.WaveSink(
            open(output_audio_file, 'wb'),
            sample_rate=audio_sample_rate,
            sample_width=audio_sample_width
        )
    else:
        audio_sink = audio_device = (
            audio_device or audio_helpers.SoundDeviceStream(
                sample_rate=audio_sample_rate,
                sample_width=audio_sample_width,
                block_size=audio_block_size,
                flush_size=audio_flush_size
            )
        )
    # Create conversation stream with the given audio source and sink.
    conversation_stream = audio_helpers.ConversationStream(
        source=audio_source,
        sink=audio_sink,
        iter_size=audio_iter_size,
        sample_width=audio_sample_width,
    )

    if not device_id or not device_model_id:
        try:
            with open(device_config) as f:
                device = json.load(f)
                device_id = device['id']
                device_model_id = device['model_id']
                logger.info("Using device model %s and device id %s",
                             device_model_id,
                             device_id)
        except Exception as e:
            logger.warning('Device config not found: %s' % e)
            logger.info('Registering device')
            if not device_model_id:
                logger.error('Option --device-model-id required '
                              'when registering a device instance.')
                sys.exit(-1)
            if not project_id:
                logger.error('Option --project-id required '
                              'when registering a device instance.')
                sys.exit(-1)
            device_base_url = (
                'https://%s/v1alpha2/projects/%s/devices' % (api_endpoint,
                                                             project_id)
            )
            device_id = str(uuid.uuid1())
            payload = {
                'id': device_id,
                'model_id': device_model_id,
                'client_type': 'SDK_SERVICE'
            }
            session = google.auth.transport.requests.AuthorizedSession(
                credentials
            )
            r = session.post(device_base_url, data=json.dumps(payload))
            if r.status_code != 200:
                logger.error('Failed to register device: %s', r.text)
                sys.exit(-1)
            logger.info('Device registered: %s', device_id)
            pathlib.Path(os.path.dirname(device_config)).mkdir(exist_ok=True)
            with open(device_config, 'w') as f:
                json.dump(payload, f)

    device_handler = device_helpers.DeviceRequestHandler(device_id)

    @device_handler.command('action.devices.commands.OnOff')
    def onoff(on):
        if on:
            logger.info('Turning device on')
        else:
            logger.info('Turning device off')

    @device_handler.command('com.example.commands.BlinkLight')
    def blink(speed, number):
        logger.info('Blinking device %s times.' % number)
        delay = 1
        if speed == "SLOWLY":
            delay = 2
        elif speed == "QUICKLY":
            delay = 0.5
        for i in range(int(number)):
            logger.info('Device is blinking.')
            time.sleep(delay)


    logger.info("Start Server")
    assistant = SampleAssistant(lang, device_model_id, device_id,
                         conversation_stream, display,
                         grpc_channel, grpc_deadline,
                         device_handler)

    # Server settings
    # Choose port 8080, for port 80, which is normally used for a http server, you need root access

    # Allow using other ports when testing
    # Local Change
    # if port:
    #     server_address = ('', int(port))
    # else:
    #     server_address = ('', 8081)
    server_address = ('', int(port))

    httpd = HTTPServer(server_address, testHTTPServer_RequestHandler)

    logger.info('Running server...')
    httpd.serve_forever()

if __name__ == '__main__':

    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--port", type=int, default=8081,
                    help="ephemeral port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())
    run(port=args["port"])
