""" ref:
https://github.com/ECI-Robotics/opencv_remote_streaming_processing/
"""

import cv2
import numpy as np
import math
import os
import sys
from logging import getLogger, basicConfig, DEBUG, INFO, ERROR
from timeit import default_timer as timer

logger = getLogger(__name__)

basicConfig(
    level=INFO,
    format="%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s")

resize_prop = (640, 480)


class VideoCamera:
    # def __init__(self, input, no_v4l, devices, cpu_extension, is_async):
    def __init__(self, input, no_v4l):
        if input == 'cam':
            self.input_stream = 0
            if no_v4l:
                self.cap = cv2.VideoCapture(self.input_stream)
            else:
                # for Picamera, added VideoCaptureAPIs(cv2.CAP_V4L)
                self.cap = cv2.VideoCapture(self.input_stream, cv2.CAP_V4L)
        else:
            self.input_stream = input
            assert os.path.isfile(input), "Specified input file doesn't exist"
            self.cap = cv2.VideoCapture(self.input_stream)

        ret, self.frame = self.cap.read()
        if ret:
            cap_prop = self._get_cap_prop()
            logger.info("cap_pop:{}, frame_prop:{}".format(
                cap_prop, resize_prop))
        else:
            logger.error(
                "Please try to start with command line parameters using --no_v4l")
            os._exit(0)

    def __del__(self):
        self.cap.release()

    def _get_cap_prop(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT), self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame(self, flip_code):

        ret, frame = self.cap.read()
        frame = cv2.resize(frame, resize_prop)

        if ret:
            if self.input_stream == 0 and flip_code is not None:
                frame = cv2.flip(frame, int(flip_code))

            return frame
