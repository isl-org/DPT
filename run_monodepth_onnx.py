import argparse
import json
import os
import glob
import numpy as np
import cv2
import onnx
import onnxruntime

import util.io


def run(input_path, output_path, model_path, kitti_crop, absolute_depth):

    model = onnx.load(model_path)
    net_w, net_h = json.loads(model.metadata_props[0].value)
    normalization = json.loads(model.metadata_props[1].value)
    prediction_factor = float(model.metadata_props[2].value)
    mean = np.array(normalization["mean"])
    std = np.array(normalization["std"])
    del model

    session = onnxruntime.InferenceSession(
        model_path,
        providers=[
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    # get input
    img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")
    for ind, img_name in enumerate(img_names):
        if os.path.isdir(img_name):
            continue

        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))
        # input

        img = util.io.read_image(img_name)

        if kitti_crop is True:
            height, width, _ = img.shape
            top = height - 352
            left = (width - 1216) // 2
            img = img[top : top + 352, left : left + 1216, :]

        # resize
        img_input = cv2.resize(img, (net_h, net_w), cv2.INTER_AREA)

        # normalize
        img_input = (img_input - mean) / std

        # transpose from HWC to CHW
        img_input = img_input.transpose(2, 0, 1)

        # add batch dimension
        img_input = img_input[None, ...]

        # compute
        prediction = session.run(["output"], {"input": img_input.astype(np.float32)})[0][0]
        prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), cv2.INTER_CUBIC)
        prediction *= prediction_factor

        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
        )
        util.io.write_depth(
            filename, prediction, bits=2, absolute_depth=absolute_depth
        )

    print("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_path", default="input", help="folder with input images"
    )

    parser.add_argument(
        "-o",
        "--output_path",
        default="output_monodepth",
        help="folder for output images",
    )

    parser.add_argument(
        "-m", "--model_weights", default=None, help="path to model weights"
    )

    parser.add_argument("--kitti_crop", dest="kitti_crop", action="store_true")
    parser.add_argument("--absolute_depth", dest="absolute_depth", action="store_true")

    parser.set_defaults(kitti_crop=False)
    parser.set_defaults(absolute_depth=False)

    args = parser.parse_args()

    # compute depth maps
    run(
        args.input_path,
        args.output_path,
        args.model_weights,
        args.kitti_crop,
        args.absolute_depth,
    )