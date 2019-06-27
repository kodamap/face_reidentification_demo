import cv2
import os
import sys
from logging import getLogger, basicConfig, DEBUG, INFO
from openvino.inference_engine import IENetwork, IEPlugin
from timeit import default_timer as timer
import numpy as np

logger = getLogger(__name__)
basicConfig(
    level=INFO,
    format="%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s")

is_myriad_plugin_initialized = False
myriad_plugin = None
is_cpu_plugin_initialized = False
cpu_plugin = None

prob_threshold_face = 0.9


class BaseDetection(object):
    def __init__(self, device, model_xml, cpu_extension, detection_of):
        """Each device's plugin should be initialized only once,
        MYRIAD plugin would be failed when createting exec_net(plugin.load method)
        Error: "RuntimeError: Can not init USB device: NC_DEVICE_NOT_FOUND"
        """

        global is_myriad_plugin_initialized
        global myriad_plugin
        global is_cpu_plugin_initialized
        global cpu_plugin

        if device == 'MYRIAD' and not is_myriad_plugin_initialized:
            self.plugin = self._init_plugin(device, cpu_extension)
            is_myriad_plugin_initialized = True
            myriad_plugin = self.plugin

        elif device == 'MYRIAD' and is_myriad_plugin_initialized:
            self.plugin = myriad_plugin

        elif device == 'CPU' and not is_cpu_plugin_initialized:
            self.plugin = self._init_plugin(device, cpu_extension)
            is_cpu_plugin_initialized = True
            cpu_plugin = self.plugin

        elif device == 'CPU' and is_cpu_plugin_initialized:
            self.plugin = cpu_plugin

        else:
            self.plugin = self._init_plugin(device, cpu_extension)

        # Read IR
        self.net = self._read_ir(model_xml, detection_of)
        # Load IR model to the plugin
        self.input_blob, self.out_blob, self.exec_net, self.input_dims, self.output_dims = self._load_ir_to_plugin(
            device, detection_of)

    def _init_plugin(self, device, cpu_extension, plugin_dir=""):

        logger.info("Initializing plugin for {} device...".format(device))
        plugin = IEPlugin(device=device, plugin_dirs=plugin_dir)
        logger.info(
            "Plugin for {} device version:{}".format(device, plugin.version))
        if cpu_extension and 'CPU' in device:
            plugin.add_cpu_extension(cpu_extension)

        return plugin

    def _read_ir(self, model_xml, detection_of):

        logger.info("Reading IR Loading for {}...".format(detection_of))
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        return IENetwork(model=model_xml, weights=model_bin)

    def _load_ir_to_plugin(self, device, detection_of):

        if device == "CPU" and detection_of == "Face Detection":

            supported_layers = self.plugin.get_supported_layers(self.net)
            not_supported_layers = [
                l for l in self.net.layers.keys() if l not in supported_layers]

            if len(not_supported_layers) != 0:

                logger.error(
                    "Following layers are not supported by the plugin for specified device {}:\n {}".
                    format(self.plugin.device, ', '.join(
                        not_supported_layers)))
                logger.error(
                    "Please try to specify cpu extensions library path in demo's command line parameters using -l "
                    "or --cpu_extension command line argument")
                sys.exit(1)

        if detection_of == "Face Detection":
            logger.info("Checking Face Detection network inputs")

            assert len(self.net.inputs.keys(
            )) == 1, "Face Detection network should have only one input"
            logger.info("Checking Face Detection network outputs")

            assert len(
                self.net.outputs) == 1, "Face Detection network should have only one output"

        input_blob = next(iter(self.net.inputs))
        out_blob = next(iter(self.net.outputs))
        logger.info("Loading {} model to the {} plugin...".format(
            device, detection_of))
        exec_net = self.plugin.load(network=self.net, num_requests=2)
        input_dims = self.net.inputs[input_blob].shape
        output_dims = self.net.outputs[out_blob].shape
        logger.info("{} input dims:{} output dims:{} ".format(
            detection_of, input_dims, output_dims))

        return input_blob, out_blob, exec_net, input_dims, output_dims


class FaceDetection(BaseDetection):
    def __init__(self, device, model_xml, cpu_extension, is_async):
        detection_of = "Face Detection"
        super().__init__(device, model_xml, cpu_extension, detection_of)

        del self.net
        self.cur_request_id = 0
        self.next_request_id = 1

    def infer(self, frame, next_frame, is_async):
        n, c, h, w = self.input_dims

        if is_async:
            logger.debug("*** start_async *** cur_req_id:{} next_req_id:{} async:{}".format(
                self.cur_request_id, self.next_request_id, is_async))
            in_frame = cv2.resize(next_frame, (w, h))
            # Change data layout from HWC to CHW
            in_frame = in_frame.transpose((2, 0, 1))
            in_frame = in_frame.reshape((n, c, h, w))
            self.exec_net.start_async(request_id=self.next_request_id, inputs={
                                      self.input_blob: in_frame})
        else:
            logger.debug("*** start_sync *** cur_req_id:{} next_req_id:{} async:{}".format(
                self.cur_request_id, self.next_request_id, is_async))
            self.exec_net.requests[self.cur_request_id].wait(-1)
            in_frame = cv2.resize(frame, (w, h))
            # Change data layout from HWC to CHW
            in_frame = in_frame.transpose((2, 0, 1))
            in_frame = in_frame.reshape((n, c, h, w))
            self.exec_net.start_async(request_id=self.cur_request_id, inputs={
                                      self.input_blob: in_frame})

    def get_results(self, is_async):
        """
        The net outputs a blob with shape: [1, 1, 200, 7]
        The description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
        """

        faces = None

        if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
            # res's shape: [1, 1, 200, 7]
            res = self.exec_net.requests[self.cur_request_id].outputs[self.out_blob]
            # Get rows whose confidence is larger than prob_threshold.
            # detected faces are also used by age/gender, emotion, landmark, head pose detection.
            faces = res[0][:, np.where(res[0][0][:, 2] > prob_threshold_face)]

        if is_async:
            self.cur_request_id, self.next_request_id = self.next_request_id, self.cur_request_id

        return faces


class FacialLandmarks(BaseDetection):
    def __init__(self, device, model_xml, cpu_extension):
        detection_of = "Facial Landmarks"
        super().__init__(device, model_xml, cpu_extension, detection_of)
        del self.net

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

        res = self.exec_net.requests[0].outputs[self.out_blob]
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
    def __init__(self, device, model_xml, cpu_extension):
        detection_of = "Face re-identifications"
        super().__init__(device, model_xml, cpu_extension, detection_of)
        del self.net

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
        res = self.exec_net.requests[0].outputs[self.out_blob]
        # save feature vectors of faces
        feature_vec = res.reshape(1, 256)
        return feature_vec
