# from model_lib.timm.model_lib.layers.helpers import to_2tuple
import torch
from model_lib import timm

def lunit_dino16(num_classes, pretrained=True):
    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=num_classes)
    if pretrained:
        td = torch.load('./model_lib/pretrained/dino_vit_small_patch16_ep200.torch')
        model.load_state_dict(td, strict=False)
    return model

def lunit_dino16_LORA8(num_classes, pretrained=True):
    model = timm.create_model('vit_small_patch16_224_LORA8', pretrained=False, num_classes=num_classes)
    if pretrained:
        td = torch.load('./model_lib/pretrained/dino_vit_small_patch16_ep200.torch')
        model.load_state_dict(td, strict=False)
    return model
