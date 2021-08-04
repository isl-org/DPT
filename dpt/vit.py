import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, List

import timm

import math

attention = {}
def get_attention(name):
    def hook(module, input, output):
        x = input[0]
        B, N, C = x.shape
        qkv = (
            module.qkv(x)
            .reshape(B, N, 3, module.num_heads, C // module.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * module.scale

        attn = attn.softmax(dim=-1)  # [:,:,1,1:]
        attention[name] = attn

    return hook


def get_mean_attention_map(attn, token, shape):
    attn = attn[:, :, token, 1:]
    attn = attn.unflatten(2, torch.Size([shape[2] // 16, shape[3] // 16])).float()
    attn = torch.nn.functional.interpolate(
        attn, size=shape[2:], mode="bicubic", align_corners=False
    ).squeeze(0)

    all_attn = torch.mean(attn, 0)

    return all_attn

class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index
    
    def forward(self, x):
        return x[:, self.start_index :]

class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)


class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    
    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)

        return self.project(features)


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


class BackboneWrapper(nn.Module):    
    def __init__(self, model, features=[256, 512, 768, 768], size=[384, 384], hooks=[0, 1, 8, 11], vit_features=768, use_vit_only=True, use_readout="ignore", start_index=1, enable_attention_hooks=False):
        super().__init__()

        self.model = model        
        self.hooks = hooks
        self.use_vit_only = use_vit_only
        
        def hook(model, input: Tuple[torch.Tensor], output: torch.Tensor):
            model.activations = output

        def add_hook(block):
            block.activations = torch.Tensor()
            block.register_forward_hook(hook)            

        if use_vit_only == True:        
            add_hook(self.model.blocks[hooks[0]])
            add_hook(self.model.blocks[hooks[1]])
        else:
            add_hook(self.model.patch_embed.backbone.stages[0])
            add_hook(self.model.patch_embed.backbone.stages[1])

        self.model.blocks[hooks[2]].activations = torch.Tensor()
        self.model.blocks[hooks[2]].register_forward_hook(hook)
        
        add_hook(self.model.blocks[hooks[3]])

        if enable_attention_hooks:
            self.model.blocks[2].attn.register_forward_hook(get_attention("attn_1"))
            self.model.blocks[5].attn.register_forward_hook(get_attention("attn_2"))
            self.model.blocks[8].attn.register_forward_hook(get_attention("attn_3"))
            self.model.blocks[11].attn.register_forward_hook(get_attention("attn_4"))
            self.attention = attention
        
        if use_vit_only == True:
            self.readout_oper1 = self.make_readout_oper(vit_features, use_readout, start_index)
            self.readout_oper2 = self.make_readout_oper(vit_features, use_readout, start_index)
        else:
            self.readout_oper1 = nn.Identity()
            self.readout_oper2 = nn.Identity()
        
        self.readout_oper3 = self.make_readout_oper(vit_features, use_readout, start_index)
        self.readout_oper4 = self.make_readout_oper(vit_features, use_readout, start_index)
        
        if use_vit_only:
            self.act_postprocess1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=vit_features,
                    out_channels=features[0],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=features[0],
                    out_channels=features[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                    bias=True,
                    dilation=1,
                    groups=1,
                ),
            )

            self.act_postprocess2 = nn.Sequential(
                nn.Conv2d(
                    in_channels=vit_features,
                    out_channels=features[1],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=features[1],
                    out_channels=features[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    bias=True,
                    dilation=1,
                    groups=1,
                ),
            )
        else:
            self.act_postprocess1 = nn.Identity()
            self.act_postprocess2 = nn.Identity()

        self.act_postprocess3 = nn.Sequential(
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[2],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        self.act_postprocess4 = nn.Sequential(
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[3],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2d(
                in_channels=features[3],
                out_channels=features[3],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

        self.start_index = start_index
        self.patch_size = [16, 16]

    def make_readout_oper(self, vit_features, use_readout: str, start_index):
        if use_readout == "ignore":
            return nn.Sequential(Slice(start_index), Transpose(1,2))
        elif use_readout == "add":
            return nn.Sequential(AddReadout(start_index), Transpose(1,2))
        elif use_readout == "project":
            return nn.Sequential(ProjectReadout(vit_features, start_index), Transpose(1,2))
        else:
            assert (
                False
            ), "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"

    def _resize_pos_embed(self, posemb: torch.Tensor, gs_h: int, gs_w: int):
        posemb_tok, posemb_grid = (
            posemb[:, : self.start_index],
            posemb[0, self.start_index :],
        )   

        gs_old = int(math.sqrt(len(posemb_grid)))

        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid, size=[gs_h, gs_w], mode="bilinear")
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

        return posemb

    def forward_flex(self, x):
        B, _, H, W = x.shape

        pos_embed = self._resize_pos_embed(
            self.model.pos_embed, int(H // self.patch_size[1]), int(W // self.patch_size[0])
        )

        B = x.shape[0]
        
        if hasattr(self.model.patch_embed, "backbone"):
            x = self.model.patch_embed.backbone(x)            
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

        x = self.model.patch_embed.proj(x).flatten(2).transpose(1, 2)
        
        if hasattr(self.model, "dist_token") and self.model.dist_token is not None:
            cls_tokens = self.model.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            dist_token = self.model.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.model.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + pos_embed
        x = self.model.pos_drop(x)

        for blk in self.model.blocks:
            x = blk(x)

        x = self.model.norm(x)

        return x

    def forward(self, x):
        _, _, h, w = x.shape

        _ = self.forward_flex(x)

        # HACK: this is to make TorchScript happy. Can't directly address modules,
        # so we gather all modules that have their activations set.
        layers = []
        if not self.use_vit_only:
            for _, v in enumerate(self.model.patch_embed.backbone.stages):
                if hasattr(v, "activations"):
                    layers.append(v.activations)

        for _, v in enumerate(self.model.blocks):
            if hasattr(v, "activations"):                
                layers.append(v.activations)

        layer_1, layer_2, layer_3, layer_4 = layers
                    
        layer_1 = self.readout_oper1(layer_1)
        layer_2 = self.readout_oper2(layer_2)        
        layer_3 = self.readout_oper3(layer_3)
        layer_4 = self.readout_oper4(layer_4)

        out_size = torch.Size((h // self.patch_size[1], w // self.patch_size[0]))

        if self.use_vit_only:
            layer_1 = self.act_postprocess1(layer_1.unflatten(2, out_size))
            layer_2 = self.act_postprocess2(layer_2.unflatten(2, out_size))
            
        layer_3 = self.act_postprocess3(layer_3.unflatten(2, out_size))
        layer_4 = self.act_postprocess4(layer_4.unflatten(2, out_size))
                
        return layer_1, layer_2, layer_3, layer_4


def _make_pretrained_vitb_rn50_384(
    pretrained,
    use_readout="ignore",
    hooks=None,
    use_vit_only=False,
    enable_attention_hooks=False,
):
    model = timm.create_model("vit_base_resnet50_384", pretrained=pretrained)

    hooks = [0, 1, 8, 11] if hooks == None else hooks
    return BackboneWrapper(
        model,
        features=[256, 512, 768, 768],
        size=[384, 384],
        hooks=hooks,
        use_vit_only=use_vit_only,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )


def _make_pretrained_vitl16_384(
    pretrained, use_readout="ignore", hooks=None, enable_attention_hooks=False
):
    model = timm.create_model("vit_large_patch16_384", pretrained=pretrained)

    hooks = [5, 11, 17, 23] if hooks == None else hooks
    return BackboneWrapper(
        model,
        features=[256, 512, 1024, 1024],
        hooks=hooks,
        vit_features=1024,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )


def _make_pretrained_vitb16_384(
    pretrained, use_readout="ignore", hooks=None, enable_attention_hooks=False
):
    model = timm.create_model("vit_base_patch16_384", pretrained=pretrained)

    hooks = [2, 5, 8, 11] if hooks == None else hooks
    return BackboneWrapper(
        model,
        features=[96, 192, 384, 768],
        hooks=hooks,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )


def _make_pretrained_deitb16_384(
    pretrained, use_readout="ignore", hooks=None, enable_attention_hooks=False
):
    model = timm.create_model("vit_deit_base_patch16_384", pretrained=pretrained)

    hooks = [2, 5, 8, 11] if hooks == None else hooks
    return BackboneWrapper(
        model,
        features=[96, 192, 384, 768],
        hooks=hooks,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )


def _make_pretrained_deitb16_distil_384(
    pretrained, use_readout="ignore", hooks=None, enable_attention_hooks=False
):
    model = timm.create_model(
        "vit_deit_base_distilled_patch16_384", pretrained=pretrained
    )

    hooks = [2, 5, 8, 11] if hooks == None else hooks
    return BackboneWrapper(
        model,
        features=[96, 192, 384, 768],
        hooks=hooks,
        use_readout=use_readout,
        start_index=2,
        enable_attention_hooks=enable_attention_hooks,
    )
