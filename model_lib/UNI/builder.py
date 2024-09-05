from model_lib import timm_v2 as timm_v2
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, ViTModel
from peft import LoraConfig, get_peft_model
import re
from collections import OrderedDict

#"hf-hub:MahmoodLab/uni"
#"vit_large_patch16_224_LORA1"
#"vit_large_patch16_224_LORA1"
#"vit_large_patch16_224_LORA1"
#"vit_large_patch16_224_LORA1"


class UNI(nn.Module):
    def __init__(self, model_name, num_classes, low_rank=0, pretrained=True):
        super().__init__()

        if low_rank > 0:
            self.uni = timm_v2.create_model(model_name, img_size=224, patch_size=16, init_values=1e-5, num_classes=0, low_rank=1, dynamic_img_size=True)
            strict = False
        else:
            self.uni = timm_v2.create_model(model_name, img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
            strict = True

        if not pretrained:
            print("Initializing uni....!!")
            self.uni.apply(self.uni.init_weights)
        else:
            pretrained_dir = 'model_lib/pretrained/pytorch_model.bin'
            msg = self.uni.load_state_dict(torch.load(pretrained_dir, map_location="cpu"), strict)
            print(msg)

        self.uni.head = nn.Linear(self.uni.embed_dim, num_classes)

    def extract_features(self, x):
        feat = self.uni.forward_features(x)
        if self.uni.attn_pool is not None:
            feat = self.attn_pool(feat)
        elif self.uni.global_pool == 'avg':
            feat = feat[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.uni.global_pool:
            feat = feat[:, 0]  # class token
        feat = self.uni.fc_norm(feat)
        feat = self.uni.head_drop(feat)

        return feat

    def forward(self, x, is_feat=False):
        feat = self.extract_features(x)
        out = self.uni.head(feat)
        if is_feat:
           return feat, out
        return out

def uni_base(num_classes, pretrained=True):
    return UNI(model_name="vit_large_patch16_224", num_classes=num_classes, pretrained=pretrained)
def uni_LORA8(num_classes, pretrained=True):
    return UNI(model_name="vit_large_patch16_224_LORA8", num_classes=num_classes, low_rank=8, pretrained=pretrained)

if __name__ == '__main__':
    import random
    import numpy as np
    import torch.backends.cudnn as cudnn

    seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = uni_base(100, True)
    model_LORA8 = uni_LORA8(100, True)

    x = model(torch.randn(1, 3, 224, 224))
    for name, param in model.named_parameters():
        print(name)
    freeze_ratio = 0.5

