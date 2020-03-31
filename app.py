###############################################################################
#
# The MIT License (MIT)
#
# Copyright (c) 2014 Miguel Grinberg
#
# Released under the MIT license
# https://github.com/miguelgrinberg/flask-video-streaming/blob/master/LICENSE
#
###############################################################################

from flask import Flask, Response, render_template, request, jsonify, url_for
from libs.camera import VideoCamera
from logging import getLogger, basicConfig, DEBUG, INFO
import os
import sys
import cv2
import numpy as np
import json
import base64
from libs.interactive_detection import Detections
from libs.register import Register
from libs.argparser import build_argparser
from libs.utils import get_frame, get_face_frames, resize_frame, cos_similarity
from datetime import datetime
from openvino.inference_engine import get_version

app = Flask(__name__)
logger = getLogger(__name__)

basicConfig(
    level=INFO,
    format="%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s")

# detection control flag
is_async = True
is_fd = False   # face detection
is_fi = False  # face re-identification
is_preprocess = False

# 0:x-axis 1:y-axis -1:both axis
flip_code = 1

search_result = {}
resize_width = 640


def preprocess(frame):
    global is_preprocess

    is_preprocess = True
    detections.face_detector.infer(frame, frame, is_async=False)
    faces = detections.face_detector.get_results(is_async=False)
    face_frames, boxes = get_face_frames(faces, frame)
    feature_vecs, aligned_faces = detections.preprocess(face_frames)
    is_preprocess = False

    return feature_vecs, aligned_faces


def gen(camera):
    while True:
        frame = camera.get_frame(flip_code)

        if not is_preprocess:
            frame = detections.face_detection(
                frame, is_async, face_vecs, face_labels, is_fd, is_fi)
        else:
            logger.info(
                "another preproces task detected: {}".format(is_preprocess))

        ret, jpeg = frame = cv2.imencode('.jpg', frame)
        frame = jpeg.tostring()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/')
def index():
    logger.info("face_labels:{}".format(face_labels))
    return render_template('index.html', is_async=is_async, flip_code=flip_code,
                           face_labels=face_labels, search_result=search_result,
                           dbname=dbname, enumerate=enumerate)


@app.route('/video_feed')
def video_feed():
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detection', methods=['POST'])
def detection():
    global is_async
    global is_fd
    global is_fi

    command = request.json['command']
    if command == "async":
        is_async = True
    elif command == "sync":
        is_async = False

    if command == "face-det":
        is_fd = not is_fd
        is_fi = False
    if command == "face-reid":
        is_fd = False
        is_fi = not is_fi

    result = {
        "command": command,
        "is_async": is_async,
        "flip_code": flip_code,
        "is_fd": is_fd,
        "is_fi": is_fi
    }
    logger.info(
        "command:{} is_async:{} flip_code:{} is_fd:{} is_fi:{}".
        format(command, is_async, flip_code, is_fd, is_fi))

    return jsonify(ResultSet=json.dumps(result))


@app.route('/flip', methods=['POST'])
def flip_frame():
    global flip_code

    command = request.json['command']

    if command == "flip" and flip_code is None:
        flip_code = 0
    elif command == "flip" and flip_code == 0:
        flip_code = 1
    elif command == "flip" and flip_code == 1:
        flip_code = -1
    elif command == "flip" and flip_code == -1:
        flip_code = None

    result = {
        "command": command,
        "is_async": is_async,
        "flip_code": flip_code
    }
    return jsonify(ResultSet=json.dumps(result))


@app.route('/registrar', methods=['POST'])
def registrar():

    # global is_async, is_fd, is_fi
    global face_labels
    global face_vecs

    command = request.json['command']

    if command == 'capture':

        frame = camera.get_frame(flip_code)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        feature_vecs, aligned_faces = preprocess(frame)

        # set ramdom label to refresh image when reloaded as same filename
        mylabel = "face" + str(datetime.now().strftime("%H%M%S"))

        face_vecs_dict, face_pics_dict = face_register.update(
            feature_vecs, aligned_faces, [mylabel])

        face_register.save(face_vecs_dict, face_pics_dict)

    if command == 'register':

        label = [request.json['label']]  # label's type is list

        data = request.json['data']
        data = base64.b64decode(data.split(',')[1])
        data = np.fromstring(data, np.uint8)

        frame = cv2.imdecode(data, cv2.IMREAD_ANYCOLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame.shape[1] >= resize_width:
            frame = resize_frame(frame, resize_width)

        feature_vecs, aligned_faces = preprocess(frame)

        face_vecs_dict, face_pics_dict = face_register.update(
            feature_vecs, aligned_faces, label)
        face_register.save(face_vecs_dict, face_pics_dict)

    if command == 'save':
        old_label = request.json['label']
        new_label = request.json['newlabel']
        logger.info("old_label:{} new_label:{}".format(old_label, new_label))

        face_vecs_dict, face_pics_dict = face_register.change(
            old_label, new_label)
        face_register.save(face_vecs_dict, face_pics_dict)

    if command == 'remove':
        label = [request.json['label']]  # label's type is list
        face_register.remove(label)

    face_labels, face_vecs = load(face_register, "face")
    logger.info("command: {}, face_labels:{}, face_vecs.shape:{}".format(
        command, face_labels, face_vecs.shape))

    result = {
        "command": command,
        "is_async": is_async,
        "flip_code": flip_code,
        "is_fd": is_fd,
        "is_fi": is_fi
    }

    return jsonify(ResultSet=json.dumps(result))


@app.route('/search', methods=['POST'])
def search():

    global search_result

    command = request.json['command']
    target_label = request.json['label']

    # get face vectors of target face
    face_vecs_dict, face_pics_dict = face_register.load()
    target_vec = face_vecs_dict[target_label]

    # similarity by descending order
    similarity = cos_similarity(target_vec, search_vecs)
    search_result = {}
    top_similarity = similarity.argsort()[::-1]

    # Return top 5 search result
    top_similarity = top_similarity[:5]

    for i, face_id in enumerate(top_similarity):
        score = "{:.2f}%".format(similarity[face_id] * 100)
        search_result[search_labels[face_id]] = score

    try:
        similarity.argmax()
        logger.info("search_result: {}".format(search_result))
        logger.info("command: {} top_similarity:{} , similarity.argmax: {}".format(
            command, top_similarity, similarity.argmax()))
    except ValueError as e:
        import traceback
        traceback.print_exc()
        logger.error(
            "Search result is empty. Check if the face database created properly.")

    result = {
        "command": command,
        "is_async": is_async,
        "flip_code": flip_code,
        "is_fd": is_fd,
        "is_fi": is_fi
    }

    return jsonify(ResultSet=json.dumps(result))


def load(obj, tag):

    face_vecs_dict, face_pics_dict = obj.load()

    logger.info("tag:{} vecs_dict:{}, pics_dict:{}".format(
        tag, len(face_vecs_dict), len(face_pics_dict)))

    face_labels = []
    face_vecs = np.zeros((len(face_vecs_dict), 256))

    for face_id, (label, face_vec) in enumerate(face_vecs_dict.items()):
        face_labels.append(label)
        face_vecs[face_id] = face_vec

    return face_labels, face_vecs


if __name__ == '__main__':

    # arg parse
    args = build_argparser().parse_args()
    devices = [args.device, args.device_landmarks,
               args.device_reidentification]
    cpu_extension = args.cpu_extension
    dbname = args.dbname

    # openvino.inference_engine version '2.1.37988' is openvino_2020.1.033 build
    # , which does not need cpu extension. 
    # https://software.intel.com/en-us/forums/intel-distribution-of-openvino-toolkit/topic/848825
    if "CPU" in devices and args.cpu_extension is None and (get_version() < '2.1.37988'):
        print(
            "\nPlugin for CPU device version is " + get_version() + " which is lower than 2.1.37988"
            "\nPlease try to specify cpu extensions library path in demo's command line parameters using -l "
            "or --cpu_extension command line argument")
        sys.exit(1)

    # load registered faces
    face_register = Register('face')
    face_labels, face_vecs = load(face_register, "face")

    # load face database for search
    searcher = Register(dbname)
    search_labels, search_vecs = load(searcher, dbname)

    logger.info("initialize face pictures. face labels:{}".format(face_labels))

    camera = VideoCamera(args.input, args.no_v4l)
    detections = Detections(camera.frame, devices, cpu_extension, is_async)

    app.run(host='0.0.0.0', threaded=True)
