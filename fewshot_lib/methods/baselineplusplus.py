import argparse
import torch
from typing import Dict, Tuple
import torch.distributed as dist
import torch.nn.functional as F

from fewshot_lib.methods.utils import get_one_hot, extract_features
from fewshot_lib.methods.method import FSmethod
import torch.nn as nn
#from torch.nn.utils.weight_norm import WeightNorm
from torch.nn.utils.parametrizations import weight_norm

class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            #WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm
            self.L = weight_norm(self.L, name='weight', dim=0)  # Apply weight normalization

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor * cos_dist

        return scores

class BaselinePlusPlus(FSmethod):
    def __init__(self, opt, feature_dim, iter=200, finetune_lr= 0.01, finetune_all_layers=False, normalize=True):
        self.feature_dim = feature_dim
        self.iter = iter
        self.lr = finetune_lr
        self.finetune_all_layers = finetune_all_layers
        self.normalize = normalize
        super().__init__(opt)

    def forward(self,
                model: torch.nn.Module,
                x_s: torch.tensor,
                x_q: torch.tensor,
                y_s: torch.tensor,
                y_q: torch.tensor,
                task_ids: Tuple[int, int] = None):
        """
        Corresponds to the TIM-GD inference
        inputs:
            x_s : torch.Tensor of shape [n_task, s_shot, feature_dim]
            x_q : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]


        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        device = x_s.device
        model.eval()
        n_tasks = x_s.size(0)
        if n_tasks > 1:
            raise ValueError('Finetune method can only deal with 1 task at a time. \
                             Currently {} tasks.'.format(n_tasks))
        y_s = y_s[0]
        #y_q = y_q[0]
        num_classes = y_s.unique().size(0)
        y_s_one_hot = get_one_hot(y_s, num_classes)

        # Initialize classifier
        classifier = distLinear(self.feature_dim, num_classes).to(device)
        # Define optimizer
        if self.finetune_all_layers:
            params = list(model.parameters()) + list(classifier.parameters())
        else:
            params = classifier.parameters()  # noqa: E127
        optimizer = torch.optim.Adam(params, lr=self.lr)

        # Run adaptation

        if self.finetune_all_layers == True:
            with torch.set_grad_enabled(True):
                for i in range(0, self.iter):
                    z_s = extract_features(x_s, model)
                    z_q = extract_features(x_q, model)
                    if self.normalize:
                        z_s = F.normalize(z_s, dim=-1)
                        z_q = F.normalize(z_q, dim=-1)
                    probs_s = classifier(z_s[0]).softmax(-1)
                loss = - (y_s_one_hot * probs_s.log()).sum(-1).mean(-1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else:
            with torch.set_grad_enabled(False):
                z_s = extract_features(x_s, model)
                z_q = extract_features(x_q, model)
                if self.normalize:
                    z_s = F.normalize(z_s, dim=-1)
                    z_q = F.normalize(z_q, dim=-1)
            for i in range(1, self.iter):
                probs_s = classifier(z_s[0]).softmax(-1)
                loss = - (y_s_one_hot * probs_s.log()).sum(-1).mean(-1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        probs_q = classifier(z_q[0]).softmax(-1).unsqueeze(0)
        return loss.detach(), probs_q.detach().argmax(2)