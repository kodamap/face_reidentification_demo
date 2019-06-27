from libs.register import Register
from libs.interactive_detection import Detections
from argparse import ArgumentParser
from libs.utils import get_frame, get_face_frames
import csv


def preprocess(image, detections):

    frame = get_frame(image)
    detections.face_detector.infer(frame, frame, is_async=False)
    faces = detections.face_detector.get_results(is_async=False)
    face_frames, boxes = get_face_frames(faces, frame)
    feature_vecs, aligned_faces = detections.preprocess(face_frames)

    return feature_vecs, aligned_faces


def create(image, detections, label):

    if image is None:
        print('Required --input.')
        return

    method = "create"
    try:
        feature_vecs, aligned_faces = preprocess(image, detections)
        face_vecs, face_pics = register.create(
            feature_vecs, aligned_faces, label)
        register.save(face_vecs, face_pics)
        register.lists(label)
    except TypeError as e:
        print(e)


def update(image, detections, label):

    if image is None:
        print('Required --input.')
        return

    method = "update"
    try:
        feature_vecs, aligned_faces = preprocess(image, detections)
        face_vecs, face_pics = register.update(
            feature_vecs, aligned_faces, label)
        register.save(face_vecs, face_pics)
        register.lists(label)
    except TypeError as e:
        print(e)


def change(label):

    old_label = label[0]
    new_label = label[1]
    face_vecs, face_pics = register.change(old_label, new_label)
    register.save(face_vecs, face_pics)

    register.lists(new_label)


def lists(label):
    register.lists(label)


def show(label):
    register.show(label)


def remove(label):
    register.remove(label)


def csv_register(csvfile, detections, batch_size):

    with open(csvfile) as csvfile:
        # Returned rows are OrderedDict (Python 3.6+)
        reader = csv.DictReader(csvfile)

        face_vecs_dict = {}
        face_pics_dict = {}

        count = 1
        for row in reader:
            image = row['imagepath']
            label = row['label']

            # cv2.error: OpenCV(4.0.0) !ssize.empty() in function 'cv::resize'
            try:
                feature_vecs, aligned_faces = preprocess(image, detections)
            except Exception as e:
                print(e)

            if len(aligned_faces) == 1:
                face_vecs, face_pics = register.create(
                    feature_vecs, aligned_faces, [label])
                face_vecs_dict.update(face_vecs)
                face_pics_dict.update(face_pics)
            else:
                print("{} faces are detected. skip {}".format(
                    len(aligned_faces), label))

            if batch_size and count % batch_size == 0:
                register.save(face_vecs_dict, face_pics_dict)
                face_vecs_dict, face_pics_dict = register.load()
                print("registered {} records".format(count))
            count += 1

    print("registered {} records".format(count))
    register.save(face_vecs_dict, face_pics_dict)


if __name__ == "__main__":

    method = ['create', 'update', 'change',
              'list', 'show', 'remove', 'csv_register']
    device = ['CPU', 'GPU', 'FPGA', 'MYRIAD']

    parser = ArgumentParser()
    parser.add_argument(
        'method',
        choices=method,
        type=str)
    parser.add_argument(
        "-i",
        "--input",
        help="Path to a image file or URL",
        type=str)
    parser.add_argument(
        "-d",
        "--device",
        help="Specify the target device for Face Detection to infer on; CPU, GPU, FPGA or MYRIAD is acceptable.",
        default="CPU",
        choices=device,
        type=str)
    parser.add_argument(
        "--label",
        nargs='+',
        help="Specify labels")
    parser.add_argument(
        "--csv",
        help="Sepecify the csv file. format: 'filepath, label, target'",
        type=str)
    parser.add_argument(
        "--dbname",
        help="db name for registration. default is 'face' ",
        default="face",
        type=str)
    parser.add_argument(
        "--batch_size",
        help="batch_size when registering face",
        default=None,
        type=int)

    args = parser.parse_args()
    devices = ['CPU', 'CPU', 'CPU']
    cpu_extension = 'extension/cpu_extension.dll'
    frame = None

    register = Register(args.dbname)

    if args.method == 'create' or args.method == 'update' or args.method == 'csv_register':
        detections = Detections(frame, devices, cpu_extension)

    if args.method == 'create':
        create(args.input, detections, args.label)

    if args.method == 'update':
        update(args.input, detections, args.label)

    if args.method == 'change':
        if args.label:
            change(args.label)
        else:
            print("No label specified. use --label old_label new_label")

    if args.method == 'list':
        lists(args.label)

    if args.method == 'show':
        show(args.label)

    if args.method == 'remove':
        if args.label:
            remove(args.label)
        else:
            print("No label specified. use --label")

    if args.method == 'csv_register':
        if args.csv:
            csv_register(args.csv, detections, args.batch_size)
        else:
            print("No csv file specified. use --csv <csvfile>")
