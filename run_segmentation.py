"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import torch.nn.functional as F
import utils
import cv2
import argparse

from torchvision.transforms import Compose
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_large
from midas.transforms import Resize, NormalizeImage, PrepareForNet


def run(input_path, output_path, model_path, model_type="large", optimize=True):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    if model_type == "vit_large":
        model = MidasNet(model_path, backbone="vitl16_384", monodepth=False, num_classes=150,
            blocks={'hooks': [5, 11, 17, 23], 'use_readout': 'project', 'activation': 'relu', 'aux': True, 'widehead': True, 'batch_norm': True}, non_negative=True)
        net_w, net_h = 480, 480
    elif model_type == "vit_hybrid":
        model = MidasNet(model_path, backbone="vitb_rn50_384", monodepth=False, num_classes=150, 
            blocks={'hooks': [0, 1, 8, 11], 'use_readout': 'project', 'activation': 'relu', 'aux': True, 'widehead': True, 'batch_norm': True}, non_negative=True)
        net_w, net_h = 480, 480
    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False
    
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
        ]
    )

    model.eval()
    
    if optimize==True:
        # rand_example = torch.rand(1, 3, net_h, net_w)
        # model(rand_example)
        # traced_script_module = torch.jit.trace(model, rand_example)
        # model = traced_script_module
    
        if device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)  
            model = model.half()

    model.to(device)

    # get input
    img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")

    for ind, img_name in enumerate(img_names):

        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))

        # input

        img = utils.read_image(img_name)
        img_input = transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
            if optimize==True and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)  
                sample = sample.half()
            out, auxout = model.forward(sample)
            
            out = F.softmax(out, dim=1)

            prediction = (
                torch.nn.functional.interpolate(
                    out,
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
            )
            prediction = torch.argmax(prediction, dim=1)
            prediction = prediction.squeeze().cpu().numpy()
            
        # output
        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
        )
        utils.write_segm_img(filename, img * 255.0, prediction, alpha=0.5)

    print("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', 
        default='input',
        help='folder with input images'
    )

    parser.add_argument('-o', '--output_path', 
        default='output',
        help='folder for output images'
    )

    parser.add_argument('-m', '--model_weights', 
        default=None,
        help='path to the trained weights of model'
    )

    # 'vit_large', 'vit_hybrid'
    parser.add_argument('-t', '--model_type', 
        default='vit_hybrid',
        help='model type: large or small'
    )

    parser.add_argument('--optimize', dest='optimize', action='store_true')
    parser.add_argument('--no-optimize', dest='optimize', action='store_false')
    parser.set_defaults(optimize=True)

    args = parser.parse_args()

    default_models = {
        'vit_large': 'seg_vit_large-b12dca68.pt', 
        'vit_hybrid': 'seg_vit_hybrid-53898607.pt', 
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]


    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(args.input_path, args.output_path, args.model_weights, args.model_type, args.optimize)
