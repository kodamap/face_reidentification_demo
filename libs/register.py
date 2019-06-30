import joblib
import sys
import matplotlib.pyplot as plt
import libs.detectors
from libs.utils import get_frame, get_face_frames, align_face, cos_similarity
import numpy as np
import cv2
import os
from logging import getLogger, basicConfig, DEBUG, INFO, ERROR

logger = getLogger(__name__)

basicConfig(
    level=INFO,
    format="%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s")


class Register:
    def __init__(self, target):
        target = str(target)
        self.images_path = 'static/images/' + target + '/'
        self.vecs_file = target + '_vecs.gz'
        self.pics_file = target + '_pics.gz'

        if not os.path.exists(self.images_path):
            os.mkdir(self.images_path)

    def load(self):

        # ordered dict (need python 3.6+)
        face_vecs_dict = {}
        face_pics_dict = {}

        if os.path.exists(self.vecs_file) and os.path.exists(self.pics_file):
            with open(self.vecs_file, 'rb') as f:
                face_vecs_dict = joblib.load(f)
            with open(self.pics_file, 'rb') as f:
                face_pics_dict = joblib.load(f)
        else:
            logger.error("{} not exists".format(self.vecs_file))

        return face_vecs_dict, face_pics_dict

    def save(self, face_vecs, face_pics):

        with open(self.vecs_file, 'wb') as f:
            joblib.dump(face_vecs, f, compress='gzip')
        with open(self.pics_file, 'wb') as f:
            joblib.dump(face_pics, f, compress='gzip')

    def create(self, feature_vecs, aligned_faces, label):
        # ordered dict (need python 3.6+)
        face_vecs_dict = {}
        face_pics_dict = {}

        for face_id, feature_vec in enumerate(feature_vecs):
            face_vecs_dict[label[face_id]] = feature_vec

        for face_id, aligned_face in enumerate(aligned_faces):
            # save image path. add '/' to user as URL Static Path
            image_file = "/" + self.images_path + label[face_id] + ".jpg"
            face_pics_dict[label[face_id]] = image_file

            # create face image files
            face_frame = cv2.cvtColor(
                aligned_faces[face_id], cv2.COLOR_BGR2RGB)
            cv2.imwrite("." + image_file, face_frame)

        return face_vecs_dict, face_pics_dict

    def update(self, feature_vecs, aligned_faces, label):
        # ordered dict (need python 3.6+)
        face_vecs = {}
        face_pics = {}
        face_vecs_dict = {}
        face_pics_dict = {}
        
        face_vecs_dict, face_pics_dict = self.load()

        # face vectors
        for face_id, feature_vec in enumerate(feature_vecs):
            try:
                if label == "":
                    face_vecs[str(face_id)] = feature_vec
                else:
                    face_vecs[label[face_id]] = feature_vec
            except IndexError:
                # when no labels are specified
                face_vecs[label[0] + str(face_id)] = feature_vec
                logger.info(
                    "Using face id to label because of lack of labels.")
            face_vecs_dict.update(face_vecs)

        # face pictures
        for face_id, aligned_face in enumerate(aligned_faces):
            try:
                image_file = "/" + self.images_path + label[face_id] + ".jpg"
                face_pics[label[face_id]] = image_file

                # create face image files
                face_frame = cv2.cvtColor(
                    aligned_faces[face_id], cv2.COLOR_BGR2RGB)
                cv2.imwrite("." + image_file, face_frame)

            except IndexError:
                # when no labels are specified
                image_file = "/" + self.images_path + \
                    label[0] + str(face_id) + ".jpg"
                face_pics[label[0] + str(face_id)] = image_file

                # create face image files
                face_frame = cv2.cvtColor(
                    aligned_faces[face_id], cv2.COLOR_BGR2RGB)
                cv2.imwrite("." + image_file, face_frame)

                logger.info(
                    "Using face id to label because of lack of labels.")

            face_pics_dict.update(face_pics)

        return face_vecs_dict, face_pics_dict

    def change(self, old_label, new_label):

        face_vecs_dict, face_pics_dict = self.load()

        if new_label in face_pics_dict:
            logger.error("new_label:[{}] already exists.".format(new_label))
            return face_vecs_dict, face_pics_dict

        old_label, new_label = str(old_label), str(new_label)

        # create new key with the data of old label (dict can not change the key)
        face_vecs_dict[new_label] = face_vecs_dict.pop(old_label)

        old_image_file = os.path.join(self.images_path, old_label + '.jpg')
        new_image_file = os.path.join(self.images_path, new_label + '.jpg')
        face_pics_dict.pop(old_label)
        face_pics_dict[new_label] = new_image_file

        self.save(face_vecs_dict, face_pics_dict)

        if os.path.exists(old_image_file):
            os.rename(old_image_file, new_image_file)
        else:
            logger.info("{} not exists".format(old_image_file))

        return face_vecs_dict, face_pics_dict

    def lists(self, labels=None):
        face_vecs, face_pics = self.load()

        if len(face_pics) == 0:
            print("No data found: vecs:{} pics:{}".format(
                len(face_pics), len(face_pics)))
            return

        if labels:
            for target_label in labels:
                if face_pics.get(target_label):
                    face_pics.get(target_label)
                    print("label:{} file:{}".format(
                        target_label, face_pics.get(target_label)))
                else:
                    print("label:{} not found".format(target_label))
        else:
            for face_id, (label, image_file) in enumerate(face_pics.items()):
                print("{}, label:{} file:{}".format(
                    face_id, label, image_file))
            print("Rows:{}".format(face_id))

    def show(self, labels=None, top=6):
        face_vecs, face_pics = self.load()
        # rows = len(face_pics) // top + 1
        rows = top if top <= len(face_pics) else len(face_pics)
        cols = top if top <= len(face_pics) else len(face_pics)
        num = rows * cols

        if labels is None:
            print("display top {} faces".format(top))
            for face_id, (label, image_file) in enumerate(face_pics.items()):
                if num >= 1 and os.path.exists("." + image_file):
                    print("{}, label:{} file:{}".format(
                        face_id, label, image_file))
                    # expect file path like "./staic/images/faces/[label].jpg"
                    frame = get_frame("." + image_file)
                    ax = plt.subplot(rows, cols, face_id + 1)
                    ax.set_title("{}".format(label), fontsize=8)
                    ax.axis('off')
                    plt.imshow(frame)
                num -= 1
            plt.show()
        else:
            for face_id, label in enumerate(labels):
                face_vec = face_vecs[label]
                image_file = face_pics[label]
                print("{}, label:{} file:{}".format(
                    face_id, label, image_file))
                if os.path.exists("." + image_file):
                    frame = get_frame("." + image_file)
                    ax = plt.subplot(rows, len(labels), face_id + 1)
                    ax.set_title("{}".format(label), fontsize=8)
                    ax.axis('off')
                    plt.imshow(frame)
            plt.show()

    def remove(self, labels):

        face_vecs, face_pics = self.load()

        # KeyError may be mistach between face_vecs and face_pics. Remove the label anyway.
        for label in labels:
            try:
                face_vecs.pop(str(label))
            except KeyError:
                logger.info(
                    "face_vecs KeyError: {} not found.".format(label))

        for label in labels:
            try:
                face_pics.pop(str(label))
            except KeyError:
                logger.info(
                    "face_pics KeyError: {} not found.".format(label))

        self.save(face_vecs, face_pics)

        image_file = os.path.join(self.images_path, label + '.jpg')

        if os.path.exists(image_file):
            os.remove(image_file)
