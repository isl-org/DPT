import cv2
import imageio
import sys
import time
import numpy as np
import os
import argparse

import torch
import torch.nn as nn

from torchvision.transforms import Compose
from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet

from tqdm import tqdm
import errno


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

parser = argparse.ArgumentParser()
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--data_path', type=str, help='path to the data', required=True)
parser.add_argument('--dataset', type=str, help='dataset: kitti or nyu', default='nyu')
parser.add_argument('--max_depth', type=float, help='maximum depth in prediction', default=80)
parser.add_argument('--do_kb_crop', help='set for kitti benchmark images', action='store_true')

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()


def get_number_lines(file_path):
    fp = open(file_path, 'r')
    lines = fp.readlines()
    fp.close()
    return len(lines)


def validate(params):
    """Validation function."""
    
    if args.dataset == 'kitti':
        
        model = MidasNet(backbone="vitb_rn50_384", features=256,
                path="./vit_hybrid_kitti-cb926ef4.pt",
                blocks={ 'expand': False, 'freeze_bn': True, 'hooks': [0, 1, 8, 11],
                    'activation': 'relu', 
                    'use_readout': 'project',
                    'scale': 0.00006016,
                    'shift': 0.00579,
                })
    else:
        model = MidasNet(backbone="vitb_rn50_384", features=256,
                path="./vit_hybrid_nyu-2ce69ec7.pt",
                blocks={ 'expand': False, 'freeze_bn': True, 'hooks': [0, 1, 8, 11],
                    'activation': 'relu', 
                    'use_readout': 'project',
                    'scale': 0.000305,
                    'shift': 0.1378,
                })

    device = torch.device("cuda")
    model.eval()
    model.to(device)

    if args.dataset == 'kitti':
        # Set up data loading
        transform = Compose(
            [
                Resize(
                    1216, 352, # Kitti test
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    # resize_method="lower_bound",
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                PrepareForNet(),
            ]
        )
    else:
        # Set up data loading
        transform = Compose(
            [
                Resize(
                    640, 480, # NYU test
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    # resize_method="lower_bound",
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                PrepareForNet(),
            ]
        )

    num_test_samples = get_number_lines(args.filenames_file)

    with open(args.filenames_file) as fp:
        lines = fp.readlines()

    print('now testing {} files '.format(num_test_samples))

  
    save_dir_name = './outputs/'
    
    print('Saving result pngs..')
    if not os.path.exists(os.path.dirname(save_dir_name)):
        print(f"Create output folder: {save_dir_name}")
        try:
            os.mkdir(save_dir_name)
            os.mkdir(save_dir_name + '/raw')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    else:
        print(f" os.path.dirname(save_dir_name) = {os.path.dirname(save_dir_name)}, save_dir_name = {save_dir_name}")

    for s in tqdm(range(num_test_samples)):
        if args.dataset == 'nyu': 
            scene_name = lines[s].split()[0].split('/')[0]
            filename_pred_png = save_dir_name + '/raw/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace('.jpg', '.png')
        elif args.dataset == 'kitti':
            date_drive = lines[s].split('/')[1]
            filename_pred_png = save_dir_name + '/raw/' + date_drive + '_' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
            focal = float(lines[s].split()[2])
        elif args.dataset == 'kitti_benchmark':
            filename_pred_png = save_dir_name + '/raw/' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
        else:
            raise SystemExit('Wrong dataset name')

        
        rgb_path = os.path.join(args.data_path, './' + lines[s].split()[0])
                
        image = np.array(imageio.imread(rgb_path, pilmode="RGB")) / 255

        if args.do_kb_crop is True:
            height = image.shape[0]
            width = image.shape[1]
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

        sample = transform({"image": image})["image"]
        sample = torch.from_numpy(sample).unsqueeze(0)
        sample = sample.to(device)

        with torch.no_grad():
            pred_depth = model(sample)

        if args.dataset == 'kitti' or args.dataset == 'kitti_benchmark':
            pred_depth = (
                torch.nn.functional.interpolate(
                    pred_depth.unsqueeze(1),
                    size=(352, 1216),
                    mode="bicubic",
                    align_corners=True,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
        else:
            pred_depth = (
                torch.nn.functional.interpolate(
                    pred_depth.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=True,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
        
        eps=1e-8
        pred_depth[pred_depth <= eps] = eps
        pred_depth = 1.0 / pred_depth

        if args.dataset == 'kitti' or args.dataset == 'kitti_benchmark':
            prediction_depth_scaled = pred_depth * 256.0
        else:
            prediction_depth_scaled = pred_depth * 1000.0
                
        prediction_depth_scaled = prediction_depth_scaled.astype(np.uint16)
        print(f" filename_pred_png = {filename_pred_png}")
        cv2.imwrite(filename_pred_png, prediction_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                
    return


if __name__ == '__main__':
    validate(args)
