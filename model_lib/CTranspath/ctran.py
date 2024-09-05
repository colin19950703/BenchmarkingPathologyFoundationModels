# from model_lib.timm.model_lib.layers.helpers import to_2tuple
from model_lib.timm.models.layers.helpers import to_2tuple
from model_lib import timm
import torch.nn as nn
import torch

class ConvStem(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten


        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

def ctranspath(num_classes, pretrained=True):
    model = timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False, num_classes=num_classes)
    if pretrained:
        td = torch.load('./model_lib/pretrained/ctranspath.pth')
        model.load_state_dict(td['model'], strict=False)

    return model

def ctranspath_LORA8(num_classes, pretrained=True):
    model = timm.create_model('swin_tiny_patch4_window7_224_LORA8', embed_layer=ConvStem, pretrained=False, num_classes=num_classes)
    if pretrained:
        td = torch.load('./model_lib/pretrained/ctranspath.pth')
        model.load_state_dict(td['model'], strict=False)
    return model