import torch
import torch.nn as nn
from transformers import AutoImageProcessor, ViTModel
from peft import LoraConfig, get_peft_model
import re
from collections import OrderedDict

class phikon(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        self.phikon = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        if not pretrained:
            print("Initializing phikon....!!")
            self.phikon.apply(self.phikon._init_weights)
        self.phikon.head = nn.Linear(self.phikon.config.hidden_size, num_classes)

    def forward(self, x, is_feat=False):
        feat = self.phikon(x)[0][:, 0]
        out = self.phikon.head(feat)

        if is_feat:
            return feat, out
        return out

def phikon_base(num_classes, pretrained=True):
    return phikon(num_classes, pretrained)

def phikon_LORA8(num_classes, pretrained=True):
    config = LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"], lora_dropout=0.1, bias="none", modules_to_save=["head"])
    model = phikon(num_classes)
    model = get_peft_model(model, config)
    return model


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

    model = phikon_base(100, False)

    for name, param in model.named_parameters():
        print(name)
    freeze_ratio = 0.5
