import torch
import argparse
import onnx
import onnxruntime
import json
import numpy as np
import cv2

from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
import util.io


def main(model_path, model_type, output_path, batch_size, test_image_path):
    # load network
    if model_type == "dpt_large":  # DPT-Large
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        prediction_factor = 1
    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        prediction_factor = 1
    elif model_type == "dpt_hybrid_kitti":
        net_w = 1216
        net_h = 352

        model = DPTDepthModel(
            path=model_path,
            scale=0.00006016,
            shift=0.00579,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        normalization = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        prediction_factor = 256
    elif model_type == "dpt_hybrid_nyu":
        net_w = 640
        net_h = 480

        model = DPTDepthModel(
            path=model_path,
            scale=0.000305,
            shift=0.1378,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        normalization = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        prediction_factor = 1000.0
    elif model_type == "midas_v21":  # Convolutional model
        net_w = net_h = 384

        model = MidasNet_large(model_path, non_negative=True)
        normalization = dict(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        prediction_factor = 1
    else:
        assert (
            False
        ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid|dpt_hybrid_kitti|dpt_hybrid_nyu|midas_v21]"

    model.eval()

    dummy_input = torch.zeros((batch_size, 3, net_h, net_w))
    # TODO: right now, the batch size is not dynamic due to the PyTorch tracer
    # treating the batch size as constant (see get_attention() in vit.py).
    # Therefore you have to use a batch size of one to use this together with
    # run_monodepth_onnx.py.
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # store normalization configuration
    model_onnx = onnx.load(output_path)
    meta_imagesize = model_onnx.metadata_props.add()
    meta_imagesize.key = "ImageSize"
    meta_imagesize.value = json.dumps([net_w, net_h])
    meta_normalization = model_onnx.metadata_props.add()
    meta_normalization.key = "Normalization"
    meta_normalization.value = json.dumps(normalization)
    meta_prediction_factor = model_onnx.metadata_props.add()
    meta_prediction_factor.key = "PredictionFactor"
    meta_prediction_factor.value = str(prediction_factor)
    onnx.save(model_onnx, output_path)
    del model_onnx

    if test_image_path is not None:
        # load test image
        img = util.io.read_image(test_image_path)

        # resize
        img_input = cv2.resize(img, (net_h, net_w), cv2.INTER_AREA)

        # normalize
        img_input = (img_input - np.array(normalization["mean"])) / np.array(normalization["std"])

        # transpose from HWC to CHW
        img_input = img_input.transpose(2, 0, 1)

        # add batch dimension
        img_input = np.stack([img_input] * batch_size)

        # validate accuracy of exported model
        torch_out = model(torch.from_numpy(img_input.astype(np.float32))).detach().cpu().numpy()
        session = onnxruntime.InferenceSession(
            output_path,
            providers=[
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        onnx_out = session.run(["output"], {"input": img_input.astype(np.float32)})[0]

        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-02, atol=1e-04)
        print("Exported model predictions match original")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_weights", help="path to input model weights")
    parser.add_argument("output_path", help="path to output model weights")
    parser.add_argument(
        "-t",
        "--model_type",
        default="dpt_hybrid",
        help="model type [dpt_large|dpt_hybrid|midas_v21]",
    )
    parser.add_argument("--batch_size", default=1, help="batch size used for tracing")
    parser.add_argument(
        "--test_image_path",
        type=str,
        help="path to some image to test the accuracy of the exported model against the original"
    )

    args = parser.parse_args()
    main(args.model_weights, args.model_type, args.output_path, args.batch_size, args.test_image_path)