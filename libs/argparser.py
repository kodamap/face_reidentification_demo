
from argparse import ArgumentParser


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="Path to video file or image. 'cam' for capturing video stream from camera",
        required=True,
        type=str)
    parser.add_argument(
        "-l",
        "--cpu_extension",
        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels impl.",
        type=str,
        default=None)
    parser.add_argument(
        "-d",
        "--device",
        help="Specify the target device for Face Detection to infer on; CPU, GPU, FPGA or MYRIAD is acceptable.",
        default="CPU",
        choices=['CPU', 'GPU', 'FPGA', 'MYRIAD'],
        type=str)
    parser.add_argument(
        "-d_lm",
        "--device_landmarks",
        help="Specify the target device for Facial Landmarks Estimation to infer on; CPU, GPU, FPGA or MYRIAD \
             is acceptable.",
        default="CPU",
        choices=['CPU', 'GPU', 'FPGA', 'MYRIAD'],
        type=str)
    parser.add_argument(
        "-d_fi",
        "--device_reidentification",
        help="Specify the target device for Facial re-identificaiton to infer on; CPU, GPU, FPGA or MYRIAD \
            is acceptable.",
        default="CPU",
        choices=['CPU', 'GPU', 'FPGA', 'MYRIAD'],
        type=str)
    parser.add_argument(
        "--dbname",
        help="Specify face database name",
        default="celeba",
        type=str)
    parser.add_argument(
        '--no_v4l',
        help='cv2.VideoCapture without cv2.CAP_V4L',
        action='store_true')
    return parser
