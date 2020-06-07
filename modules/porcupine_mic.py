#
# Copyright 2018 Picovoice Inc.
#
# You may not use this file except in compliance with the license. A copy of the license is located in the "LICENSE"
# file accompanying this source.
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
#

import os
import struct
from datetime import datetime
from threading import Thread

import numpy as np
import pyaudio
import soundfile
import pvporcupine

from pvporcupine import Porcupine

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s %(message)s')


class PorcupineDemo(Thread):
    """
    Demo class for wake word detection (aka Porcupine) library. It creates an input audio stream from a microphone,
    monitors it, and upon detecting the specified wake word(s) prints the detection time and index of wake word on
    console. It optionally saves the recorded audio into a file for further review.
    """

    def __init__(
            self,
            library_path=pvporcupine.LIBRARY_PATH,
            model_file_path=pvporcupine.MODEL_FILE_PATH,
            keywords="porcupine,bumblebee,grasshopper",
            keyword_file_paths=None,
            sensitivities=0.5,
            input_device_index=0,
            output_path=None):

        """
        Constructor.

        :param library_path: Absolute path to Porcupine's dynamic library.
        :param model_file_path: Absolute path to the model parameter file.
        :param keyword_file_paths: List of absolute paths to keyword files.
        :param sensitivities: Sensitivity parameter for each wake word. For more information refer to
        'include/pv_porcupine.h'. It uses the
        same sensitivity value for all keywords.
        :param input_device_index: Optional argument. If provided, audio is recorded from this input device. Otherwise,
        the default audio input device is used.
        :param output_path: If provided recorded audio will be stored in this location at the end of the run.
        """

        super(PorcupineDemo, self).__init__()
        KEYWORD_FILE_PATHS = pvporcupine.KEYWORD_FILE_PATHS

        if keyword_file_paths is None:
            if keywords is None:
                raise ValueError('either --keywords or --keyword_file_paths must be set')

            keywords = [x.strip() for x in keywords.split(',')]

            if all(x in pvporcupine.KEYWORDS for x in keywords):
                keyword_file_paths = [KEYWORD_FILE_PATHS[x] for x in keywords]
            else:
                raise ValueError(
                    'selected keywords are not available by default. available keywords are: %s' % ', '.join(pvporcupine.KEYWORDS))
        else:
            keyword_file_paths = [x.strip() for x in keyword_file_paths.split(',')]

        if isinstance(sensitivities, float):
            sensitivities = [sensitivities] * len(keyword_file_paths)
        else:
            sensitivities = [float(x) for x in sensitivities.split(',')]

        self._library_path = library_path
        self._model_file_path = model_file_path
        self._keyword_file_paths = keyword_file_paths
        self._sensitivities = sensitivities
        self._input_device_index = input_device_index
        self.detected_time = datetime(2000,1,1)

        self._output_path = output_path
        if self._output_path is not None:
            self._recorded_frames = []

    def reset_detected(self):
        self.detected_time = datetime(2000,1,1)

    def run(self):
        """
         Creates an input audio stream, initializes wake word detection (Porcupine) object, and monitors the audio
         stream for occurrences of the wake word(s). It prints the time of detection for each occurrence and index of
         wake word.
         """

        num_keywords = len(self._keyword_file_paths)

        keyword_names = list()
        for x in self._keyword_file_paths:
            keyword_names.append(os.path.basename(x).replace('.ppn', '').replace('_compressed', '').split('_')[0])

        logger.info('listening for:')
        for keyword_name, sensitivity in zip(keyword_names, self._sensitivities):
            logger.info('- %s (sensitivity: %.2f)' % (keyword_name, sensitivity))

        porcupine = None
        pa = None
        audio_stream = None
        try:
            porcupine = Porcupine(
                library_path=self._library_path,
                model_file_path=self._model_file_path,
                keyword_file_paths=self._keyword_file_paths,
                sensitivities=self._sensitivities)

            pa = pyaudio.PyAudio()
            audio_stream = pa.open(
                rate=porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=porcupine.frame_length,
                input_device_index=self._input_device_index)

            # start_time = datetime.now()
            # delta = start_time - start_time
            result = None

            #while delta.seconds < run_time and not result:
            while True:
                pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow = False)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

                if self._output_path is not None:
                    self._recorded_frames.append(pcm)

                result = porcupine.process(pcm)
                if num_keywords == 1 and result:
                    logger.info('detected keyword')
                    self.detected_time = datetime.now()
                elif num_keywords > 1 and result >= 0:
                    logger.info('detected keyword: %s' % (keyword_names[result]))
                    self.detected_time = datetime.now()
                    self.detected_keyword = keyword_names[result]

                # now_time = datetime.now()
                # delta = now_time - start_time

        except KeyboardInterrupt:
            logger.info('stopping ...')
        finally:
            logger.info('Timeout or wake word detected. Cleaning up...')
            if porcupine is not None:
                porcupine.delete()

            if audio_stream is not None:
                audio_stream.close()

            if pa is not None:
                pa.terminate()

            if self._output_path is not None and len(self._recorded_frames) > 0:
                recorded_audio = np.concatenate(self._recorded_frames, axis=0).astype(np.int16)
                soundfile.write(self._output_path, recorded_audio, samplerate=porcupine.sample_rate, subtype='PCM_16')

        return num_keywords

