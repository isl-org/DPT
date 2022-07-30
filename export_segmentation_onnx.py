import torch
import argparse
import onnx
import onnxruntime
import json
import numpy as np
import cv2

from dpt.models import DPTSegmentationModel
import util.io


def main(model_path, model_type, output_path, batch_size, test_image_path):
    net_w = net_h = 480
    normalization = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # load network
    if model_type == "dpt_large":
        model = DPTSegmentationModel(
            150,
            path=model_path,
            backbone="vitl16_384",
        )
    elif model_type == "dpt_hybrid":
        model = DPTSegmentationModel(
            150,
            path=model_path,
            backbone="vitb_rn50_384",
        )
    else:
        assert (
            False
        ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid]"

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
        help="model type [dpt_large|dpt_hybrid]",
    )
    parser.add_argument("--batch_size", default=1, help="batch size used for tracing")
    parser.add_argument(
        "--test_image_path",
        type=str,
        help="path to some image to test the accuracy of the exported model against the original"
    )

    args = parser.parse_args()
    main(args.model_weights, args.model_type, args.output_path, args.batch_size, args.test_image_path)