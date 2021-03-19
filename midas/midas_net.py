"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import torch
import torch.nn as nn

from .base_model import BaseModel
from .blocks import FeatureFusionBlock, FeatureFusionBlock_custom, Interpolate, _make_encoder, forward_vit



class MidasNet(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=256, backbone="vitb_rn50_384", non_negative=True, exportable=False, channels_last=False, align_corners=True,
        blocks={
            'activation': 'relu', 
            'freeze_bn': True,
            'expand': False,
            'hooks': [0, 1, 8, 11],
            'use_readout': 'project',
            'scale': 1.0,
            'shift': 0.0,
            }):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(MidasNet, self).__init__()

        use_pretrained = False if path else True
                
        self.channels_last = channels_last
        self.blocks = blocks
        self.backbone = backbone

        self.groups = 1

        self.hooks = None
        if "hooks" in self.blocks:
            self.hooks = self.blocks['hooks']

        self.use_readout="ignore"
        if "use_readout" in self.blocks:
            self.use_readout = self.blocks['use_readout']
            
        self.scale = 1.0
        if ('scale' in self.blocks):
            self.scale = self.blocks['scale']

        self.shift = 0.0
        if ('shift' in self.blocks):
            self.shift = self.blocks['shift']

        self.features1=features
        self.features2=features
        self.features3=features
        self.features4=features

        features1=features
        features2=features
        features3=features
        features4=features
        self.expand = False
        if "expand" in self.blocks and self.blocks['expand'] == True:
            self.expand = True
            features1=features
            features2=features*2
            features3=features*4
            features4=features*8

        self.pretrained, self.scratch = _make_encoder(self.backbone, features, use_pretrained, groups=self.groups, 
            expand=self.expand, exportable=exportable, hooks = self.hooks, use_readout=self.use_readout)

        print(" features(1,2,3,4) = ", features1, features2, features3, features4)

        if "activation" not in self.blocks:
            blocks['activation'] = None

        if (blocks['activation'] == 'mish'):
            self.scratch.activation = Mish()
        elif (blocks['activation'] == 'hard_mish'):
            self.scratch.activation = HardMish()
        elif (blocks['activation'] == 'leaky'):
            self.scratch.activation = nn.LeakyReLU(0.1)
        elif (blocks['activation'] == 'relu'):
            self.scratch.activation = nn.ReLU(False)
        else:
            self.scratch.activation = nn.Identity()

        self.scratch.refinenet4 = FeatureFusionBlock_custom(features4, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet3 = FeatureFusionBlock_custom(features3, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet2 = FeatureFusionBlock_custom(features2, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(features1, self.scratch.activation, deconv=False, bn=False, align_corners=align_corners)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features//2, kernel_size=3, stride=1, padding=1, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features//2, 32, kernel_size=3, stride=1, padding=1),
            self.scratch.activation,
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )
        
        if path:
            self.load(path)

        self.freeze_bn = True
        if "freeze_bn" in self.blocks and self.blocks['freeze_bn'] == False:
            self.freeze_bn = False

        if self.freeze_bn == True:
            for m in self.pretrained.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                    m.eval()
                    if True:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
                        

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        if self.channels_last==True:
            print("self.channels_last = ", self.channels_last)
            x.contiguous(memory_format=torch.channels_last)

        if hasattr(self.pretrained, 'model'):
            if hasattr(self.pretrained.model, 'patch_size'):
                # Use resizable ViT
                layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)
        else:
            layer_1 = self.pretrained.layer1(x)
            layer_2 = self.pretrained.layer2(layer_1)
            layer_3 = self.pretrained.layer3(layer_2)
            layer_4 = self.pretrained.layer4(layer_3)


        layer_1_rn = self.scratch.layer1_rn(layer_1)     
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # print(path_1.shape)

        out = self.scratch.output_conv(path_1)

        out = out * self.scale + self.shift

        return torch.squeeze(out, dim=1)


