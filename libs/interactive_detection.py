from logging import getLogger, basicConfig, DEBUG, INFO
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from libs.utils import get_face_frames, align_face, cos_similarity
import libs.detectors as detectors
import pickle as pkl

logger = getLogger(__name__)
basicConfig(
    level=INFO,
    format="%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s")

FP32 = "extension/IR/FP32/"
FP16 = "extension/IR/FP16/"

model_fc_xml = 'face-detection-retail-0004.xml'  # input , shape: [1x3x300x300]
# model_fc_xml = 'face-detection-adas-0001.xml'   # input , shape: [1x3x384x672]
model_lm_xml = 'landmarks-regression-retail-0009.xml'
model_fi_xml = 'face-reidentification-retail-0095.xml'

# Threshold of similarity to draw result on faces
sim_threshold = 0.4
# Limit count to infer face reidentification
fi_limit = 4


class Detectors:
    def __init__(self, devices, cpu_extension, is_async):

        self.cpu_extension = cpu_extension
        self.device_fc, self.device_lm, self.device_fi = devices
        self.is_async = is_async
        self._define_models()
        self._load_detectors()
        self.colors = pkl.load(open("pallete", "rb"))

    def _define_models(self):

        # set devices and models
        fp_path = FP32 if self.device_fc == "CPU" else FP16
        self.model_fc = fp_path + model_fc_xml

        fp_path = FP32 if self.device_lm == "CPU" else FP16
        self.model_lm = fp_path + model_lm_xml

        fp_path = FP32 if self.device_fi == "CPU" else FP16
        self.model_fi = fp_path + model_fi_xml

    def _load_detectors(self):

        # face_detection
        self.face_detector = detectors.FaceDetection(
            self.device_fc, self.model_fc, self.cpu_extension, self.is_async)

        # facial_landmark
        self.landmarks_detector = detectors.FacialLandmarks(
            self.device_lm, self.model_lm, self.cpu_extension)

        # face re-identification
        self.face_id_detector = detectors.FaceReIdentification(
            self.device_fi, self.model_fi, self.cpu_extension)


class Detections(Detectors):
    def __init__(self, frame, devices, cpu_extension, is_async=True):
        super().__init__(devices, cpu_extension, is_async)

        # initialize Calculate FPS
        self.accum_time = 0
        self.curr_fps = 0
        self.fps = "FPS: ??"
        self.prev_time = timer()
        self.frame = frame

    def preprocess(self, face_frames):

        # 1. get landmarks
        facial_landmarks_per_face = []

        for face_id, face_frame in enumerate(face_frames):
            lm_face = face_frame.copy()
            self.landmarks_detector.infer(lm_face)
            facial_landmarks = self.landmarks_detector.get_results(lm_face)
            facial_landmarks_per_face.append(facial_landmarks)

        # 2. align faces
        aligned_faces = []

        for face_id, face_frame in enumerate(face_frames):
            aligned_face = face_frame.copy()
            aligned_face = align_face(
                aligned_face, facial_landmarks_per_face[face_id])
            aligned_faces.append(aligned_face)

        # 3. get feature vectors of faces
        feature_vecs = np.zeros((len(aligned_faces), 256))

        for face_id, aligned_face in enumerate(aligned_faces):
            self.face_id_detector.infer(aligned_face)
            feature_vec = self.face_id_detector.get_results()
            feature_vecs[face_id] = feature_vec

        return feature_vecs, aligned_faces

    def face_detection(self, frame, is_async, face_vecs, face_labels, is_fc, is_fi):

        logger.debug("is_async:{}, is_fc:{}, is_fi:{}".format(
            is_async, is_fc, is_fi))

        green = (0, 255, 0)
        skyblue = (255, 255, 0)
        det_time = 0
        det_time_fc = 0
        det_time_fi = 0

        # just return frame when face detection and face reidentification are False
        if not is_fc and not is_fi:
            frame = self.draw_perf_stats(
                det_time, "Video capture mode", frame, is_async)
            return frame

        # ----------- Face Detection ---------- #
        logger.debug("** face_detection start **")

        if is_async:
            next_frame = frame
        else:
            next_frame = None
            self.frame = frame

        inf_start = timer()
        self.face_detector.infer(self.frame, next_frame, is_async)
        faces = self.face_detector.get_results(is_async)
        inf_end = timer()

        # check faces
        if faces is None or faces.shape[2] == 0:
            logger.info("no faces detected.")
            frame = self.draw_perf_stats(
                det_time, "No faces are detected", frame, is_async)
            return frame

        face_frames, boxes = get_face_frames(faces, self.frame)

        det_time_fc = inf_end - inf_start
        det_time_txt = "face det:{:.3f} ms ".format(det_time_fc * 1000)

        # Resizing face_frame will be failed when witdh or height of the face_fame is 0 ex. (243, 0, 3)
        for face_frame in face_frames:
            face_w, face_h = face_frame.shape[:2]
            if face_w == 0 or face_h == 0:
                logger.info(
                    "Unexpected face frame shape. face_h:{} face_w:{}".
                    format(face_h, face_w))
                return frame

        # Draw box and confidence
        if is_fc:
            for face_id, face_frame in enumerate(face_frames):
                # get box of each face
                xmin, ymin, xmax, ymax = boxes[face_id]
                confidence = round(faces[0][0][face_id][2] * 100, 1)
                result = str(face_id) + " " + str(confidence) + '%'

                cv2.rectangle(self.frame, (xmin, ymin - 22),
                              (xmax, ymin), green, -1)
                cv2.rectangle(self.frame, (xmin, ymin - 22),
                              (xmax, ymin), (255, 255, 255))
                cv2.rectangle(self.frame, (xmin, ymin),
                              (xmax, ymax), green, 1)
                cv2.putText(self.frame, result, (xmin + 3, ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
                logger.debug("face_id:{} confidence:{}%".format(
                    face_id, confidence))

        # ----------- Face re-identification ---------- #
        if is_fi and face_vecs.any():
            logger.debug("** re-identification start **")

            # set target
            inf_start = timer()
            # select 'fi_limit' faces. Too many faces effect performance.
            feature_vecs, aligned_faces = self.preprocess(
                face_frames[:fi_limit])
            inf_end = timer()

            det_time_fi = inf_end - inf_start
            det_time_txt = det_time_txt + \
                "reid:{:.3f} ms ".format(det_time_fi * 1000)

            # get similarity per face feature vectors
            for i, target_vec in enumerate(feature_vecs):
                similarity = cos_similarity(target_vec, face_vecs)

                # get index of the most similar face
                face_id = similarity.argmax()
                logger.debug("similarity:{} , similarity.argmax: {}".format(
                    similarity, similarity.argmax()))
                xmin, ymin, xmax, ymax = boxes[i]
                score = round(similarity[face_id] * 100, 1)
                result = face_labels[face_id] + " " + str(score) + '%'
                size = cv2.getTextSize(
                    result, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                xtext = xmin + size[0][0] + 20

                # draw similarity of each face into the frame
                if similarity[face_id] > sim_threshold:
                    cv2.rectangle(self.frame, (xmin, ymin - 22),
                                  (xtext, ymin), self.colors[face_id], -1)
                    cv2.rectangle(self.frame, (xmin, ymin - 22),
                                  (xtext, ymin), self.colors[face_id])
                    cv2.rectangle(self.frame, (xmin, ymin),
                                  (xmax, ymax), self.colors[face_id], 1)
                    cv2.putText(self.frame, result, (xmin + 3, ymin - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        det_time = det_time_fc + det_time_fi
        frame = self.draw_perf_stats(
            det_time, det_time_txt, self.frame, is_async)

        if is_async:
            self.frame = next_frame

        return frame

    def draw_perf_stats(self, det_time, det_time_txt, frame, is_async):

        # Draw FPS in top left corner
        fps = self.calc_fps()
        cv2.rectangle(frame, (frame.shape[1] - 50, 0), (frame.shape[1], 17),
                      (255, 255, 255), -1)
        cv2.putText(frame, fps, (frame.shape[1] - 50 + 3, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        # Draw performance stats
        if is_async:
            inf_time_message = "Total Inference time: {:.3f} ms for async mode".format(
                det_time * 1000)
        else:
            inf_time_message = "Total Inference time: {:.3f} ms for sync mode".format(
                det_time * 1000)
        cv2.putText(frame, inf_time_message, (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 10, 10), 1)
        if det_time_txt:
            inf_time_message_each = "Detection time: {}".format(det_time_txt)
            cv2.putText(frame, inf_time_message_each, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 10, 10), 1)
        return frame

    def calc_fps(self):
        curr_time = timer()
        exec_time = curr_time - self.prev_time
        self.prev_time = curr_time
        self.accum_time = self.accum_time + exec_time
        self.curr_fps = self.curr_fps + 1

        if self.accum_time > 1:
            self.accum_time = self.accum_time - 1
            self.fps = "FPS: " + str(self.curr_fps)
            self.curr_fps = 0

        return self.fps
