import cv2
import os
import sys
from logging import getLogger, basicConfig, DEBUG, INFO

from openvino.inference_engine import IECore

# OpenVINO 2021: The IEPlugin class is deprecated
try:
    from openvino.inference_engine import IEPlugin
except ImportError:
    pass
from openvino.inference_engine import get_version

import numpy as np

logger = getLogger(__name__)
basicConfig(
    level=INFO, format="%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s"
)

is_myriad_plugin_initialized = False
myriad_plugin = None

prob_threshold_face = 0.8


class BaseDetection(object):
    def __init__(self, device, model_xml, detection_of):
        self.ie = IECore()
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        print(model_xml, model_bin)
        net = self.ie.read_network(model=model_xml, weights=model_bin)
        # Load IR model to the plugin
        logger.info("Reading IR for {}...".format(detection_of))
        self._load_ir_to_plugin(device, net, detection_of)

    def _load_ir_to_plugin(self, device, net, detection_of):
        """MYRIAD device's plugin should be initialized only once,
        MYRIAD plugin would be failed when creating exec_net
        RuntimeError: Can not init Myriad device: NC_ERROR
        """

        global is_myriad_plugin_initialized
        global myriad_plugin

        if detection_of == "Face Detection":
            logger.info(f"Checking {detection_of} network inputs")
            assert len(net.input_info.keys()
                       ) == 1, "network should have only one input"
            assert len(net.outputs) == 1, "network should have only one output"

        self.input_blob = next(iter(net.input_info))
        self.out_blob = next(iter(net.outputs))

        # Loading model to the plugin
        logger.info(
            f"Loading {device} model to the {detection_of} plugin... version:{get_version()}"
        )
        # Example: version: 2021.4.2-3974-e2a469a3450-releases/2021/4
        # The IEPlugin class is deprecated
        if str(get_version().split("-")[0]) > "2021":
            self.exec_net = self.ie.load_network(
                network=net, device_name=device, num_requests=2
            )
        else:
            if device == "MYRIAD" and not is_myriad_plugin_initialized:
                # To prevent MYRIAD Plugin from initializing failed, use IEPlugin Class which is deprecated
                # "RuntimeError: Can not init Myriad device: NC_ERROR"
                self.plugin = IEPlugin(device=device, plugin_dirs=None)
                self.exec_net = self.plugin.load(network=net, num_requests=2)
                is_myriad_plugin_initialized = True
                myriad_plugin = self.plugin
            elif device == "MYRIAD" and is_myriad_plugin_initialized:
                logger.info(f"device plugin for {device} already initialized")
                self.plugin = myriad_plugin
                self.exec_net = self.plugin.load(network=net, num_requests=2)
            else:
                self.exec_net = self.ie.load_network(
                    network=net, device_name=device, num_requests=2
                )

        self.input_dims = net.input_info[self.input_blob].input_data.shape
        self.output_dims = net.outputs[self.out_blob].shape
        logger.info(
            f"{detection_of} input dims:{self.input_dims} output dims:{self.output_dims}"
        )


class FaceDetection(BaseDetection):
    def __init__(self, device, model_xml):
        detection_of = "Face Detection"
        super().__init__(device, model_xml, detection_of)

        self.cur_request_id = 0
        self.next_request_id = 1

    def infer(self, frame, next_frame, is_async):
        n, c, h, w = self.input_dims

        if is_async:
            logger.debug(
                "*** start_async *** cur_req_id:{} next_req_id:{} async:{}".format(
                    self.cur_request_id, self.next_request_id, is_async
                )
            )
            in_frame = cv2.resize(next_frame, (w, h))
            # Change data layout from HWC to CHW
            in_frame = in_frame.transpose((2, 0, 1))
            in_frame = in_frame.reshape((n, c, h, w))
            self.exec_net.start_async(
                request_id=self.next_request_id, inputs={
                    self.input_blob: in_frame}
            )
        else:
            logger.debug(
                "*** start_sync *** cur_req_id:{} next_req_id:{} async:{}".format(
                    self.cur_request_id, self.next_request_id, is_async
                )
            )
            self.exec_net.requests[self.cur_request_id].wait(-1)
            in_frame = cv2.resize(frame, (w, h))
            # Change data layout from HWC to CHW
            in_frame = in_frame.transpose((2, 0, 1))
            in_frame = in_frame.reshape((n, c, h, w))
            self.exec_net.start_async(
                request_id=self.cur_request_id, inputs={
                    self.input_blob: in_frame}
            )

    def get_results(self, is_async):
        """
        The net outputs a blob with shape: [1, 1, 200, 7]
        The description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
        """

        faces = None

        if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
            # res's shape: [1, 1, 200, 7]
            res = self.exec_net.requests[self.cur_request_id].output_blobs[self.out_blob].buffer
            # Get rows whose confidence is larger than prob_threshold.
            # detected faces are also used by age/gender, emotion, landmark, head pose detection.
            faces = res[0][:, np.where(res[0][0][:, 2] > prob_threshold_face)]

        if is_async:
            self.cur_request_id, self.next_request_id = (
                self.next_request_id,
                self.cur_request_id,
            )

        return faces


class FacialLandmarks(BaseDetection):
    def __init__(self, device, model_xml):
        detection_of = "Facial Landmarks"
        super().__init__(device, model_xml, detection_of)

    def infer(self, face_frame):
        n, c, h, w = self.input_dims
        in_frame = cv2.resize(face_frame, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape((n, c, h, w))
        self.exec_net.infer(inputs={self.input_blob: in_frame})

    def get_results(self, lm_face):
        """
        landmarks-regression-retail-0009:
          "95", [1, 10, 1, 1], containing a row-vector of 10 floating point values for five landmarks
                coordinates in the form (x0, y0, x1, y1, ..., x5, y5).
                All the coordinates are normalized to be in range [0,1]
        five keypoints (left eye, right eye, tip of nose, left lip corner, right lip corner)
        """

        # res = self.exec_net.requests[0].outputs[self.out_blob]
        res = self.exec_net.requests[0].output_blobs[self.out_blob].buffer
        res = res.reshape(1, 10)[0]  # (10,)

        facial_landmarks = np.zeros((5, 2))  # five keypoints (x, y)
        for i in range(res.size // 2):
            normed_x = res[2 * i]
            normed_y = res[2 * i + 1]
            x_lm = lm_face.shape[1] * normed_x
            y_lm = lm_face.shape[0] * normed_y
            facial_landmarks[i] = (x_lm, y_lm)
        return facial_landmarks


class FaceReIdentification(BaseDetection):
    def __init__(self, device, model_xml):
        detection_of = "Face re-identifications"
        super().__init__(device, model_xml, detection_of)

    def infer(self, aligned_face):
        n, c, h, w = self.input_dims
        in_frame = cv2.resize(aligned_face, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape((n, c, h, w))
        self.exec_net.infer(inputs={self.input_blob: in_frame})

    def get_results(self):
        """
        face-reidentification-retail-0095:
          "658", [1, 256, 1, 1], containing a row-vector of 10 floating point values for five landmarks
        """
        # res = self.exec_net.requests[0].outputs[self.out_blob]
        res = self.exec_net.requests[0].output_blobs[self.out_blob].buffer
        # save feature vectors of faces
        feature_vec = res.reshape(1, 256)
        return feature_vec
