import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)


class DPT(BaseModel):
    def __init__(
            self,
            head,
            path=None,
            features=256,
            backbone="vitb_rn50_384",
            readout="project",
            channels_last=False,
            use_bn=False):

        super(DPT, self).__init__()

        use_pretrained = False if path else True

        self.channels_last = channels_last


        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            use_pretrained,
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )

        self.scratch.refinenet4 = FeatureFusionBlock_custom(
            features,
            nn.ReLU(False),
            deconv=False,
            bn=use_bn,
            expand=False,
            align_corners=True,
        )

        self.scratch.refinenet3 = FeatureFusionBlock_custom(
            features,
            nn.ReLU(False),
            deconv=False,
            bn=use_bn,
            expand=False,
            align_corners=True,
        )

        self.scratch.refinenet2 = FeatureFusionBlock_custom(
            features,
            nn.ReLU(False),
            deconv=False,
            bn=use_bn,
            expand=False,
            align_corners=True,
        )

        self.scratch.refinenet1 = FeatureFusionBlock_custom(
            features,
            nn.ReLU(False),
            deconv=False,
            bn=use_bn,
            expand=False,
            align_corners=True,
        )

        self.scratch.output_conv = head

        if path:
            self.load(path)


    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return out


class DPTDepthModel(DPT):
    def __init__(self, **kwargs):
        features = kwargs["features"] if "features" in kwargs else 256

        non_negative = kwargs["non_negative"]
        del kwargs["non_negative"]

        head = nn.Sequential(
                nn.Conv2d(
                    features,
                    features // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                Interpolate(scale_factor=2, mode="bilinear"),
                nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True) if non_negative else nn.Identity(),
                nn.Identity(),
            )

        super().__init__(head, **kwargs)

    def forward(self, x):
        return super().forward(x).squeeze(dim=1)


class DPTSegmentationModel(DPT):
    def __init__(self, num_classes, **kwargs):

       features = kwargs["features"] if "features" in kwargs else 256

       num_classes = kwargs["num_classes"]
       del kwargs["num_classes"]

       head = nn.Sequential(
                    nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(features),
                    nn.ReLU(True),
                    nn.Dropout(0.1, False),
                    nn.Conv2d(features, num_classes, kernel_size=1),
                    Interpolate(
                        scale_factor=2, mode="bilinear", align_corners=True
                    ),
                )

       super().__init__(head, **kwargs)
