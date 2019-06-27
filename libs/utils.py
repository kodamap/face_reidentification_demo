import requests
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import re
import traceback


def get_frame(image):

    resize_width = 640

    if re.match(r'http.?://', image):
        response = requests.get(image)
        frame = np.array(Image.open(BytesIO(response.content)))
    else:
        frame = cv2.imread(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame.shape[1] >= resize_width:
        frame = resize_frame(frame, resize_width)

    return frame


def get_face_frames(faces, frame):

    frame_h, frame_w = frame.shape[:2]
    face_frames = []
    boxes = []
    resize_width = 120

    for face_id, face in enumerate(faces[0][0]):
        box = face[3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
        (xmin, ymin, xmax, ymax) = box.astype("int")
        boxes.append((xmin, ymin, xmax, ymax))
        face_frame = frame[ymin:ymax, xmin:xmax]
        face_frame = resize_frame(face_frame, resize_width)
        face_frames.append(face_frame)

    return face_frames, boxes


def resize_frame(frame, height):

    try:
        scale = height / frame.shape[1]
    except ZeroDivisionError as e:
        traceback.print_exc()
        return frame
    try:
        frame = cv2.resize(frame, dsize=None, fx=scale, fy=scale)
    except cv2.error as e:
        traceback.print_exc()

    return frame


def align_face(face_frame, landmarks):

    # ref: https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/

    left_eye, right_eye, tip_of_nose, left_lip, right_lip = landmarks

    # compute the angle between the eye centroids
    dy = right_eye[1] - left_eye[1]     # right eye, left eye Y
    dx = right_eye[0] - left_eye[0]  # right eye, left eye X
    angle = np.arctan2(dy, dx) * 180 / np.pi
    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    ##eyes_center = ((right_eye[0] + left_eye[0]) // 2, (right_eye[1] + left_eye[1]) // 2)

    # center of face_frame
    center = (face_frame.shape[0] // 2, face_frame.shape[1] // 2)
    h, w, c = face_frame.shape

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    aligned_face = cv2.warpAffine(face_frame, M, (w, h))

    return aligned_face


def cos_similarity(X, Y):
    Y = Y.T    # (1, 256) x (256, n) = (1, n)
    return np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y, axis=0))
