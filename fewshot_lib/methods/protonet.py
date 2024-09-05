import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from fewshot_lib.methods.utils import compute_centroids, extract_features, l2_distance_to_prototypes
from fewshot_lib.methods.utils import compute_centroids, extract_features, l2_distance_to_prototypes
from fewshot_lib.methods.method import FSmethod


class ProtoNet(FSmethod):
    '''
    ProtoNet method
    '''

    def __init__(self, opt: argparse.Namespace):
        super().__init__(opt)

    def forward(self,
                x_s: torch.tensor,
                x_q: torch.tensor,
                y_s: torch.tensor,
                y_q: torch.tensor,
                model: nn.Module):
        """
        inputs:
            x_s : torch.Tensor of size [s_shot, c, h, w]
            x_q : torch.Tensor of size [q_shot, c, h, w]
            y_s : torch.Tensor of size [s_shot]
            y_q : torch.Tensor of size [q_shot]
        """
        num_classes = y_s.unique().size(0)
        if not self.training:
            with torch.no_grad():
                z_s = extract_features(x_s, model)
                z_q = extract_features(x_q, model)
        else:
            z_s = extract_features(x_s, model)
            z_q = extract_features(x_q, model)

        centroids = compute_centroids(z_s, y_s)  # [batch, num_class, d]

        scores = l2_distance_to_prototypes(z_q, centroids)
        #log_probas = scores.softmax(-1)  # [batch, q_shot, num_class]
        #one_hot_q = get_one_hot(y_q, num_classes)  # [batch, q_shot, num_class]
        #ce = -(one_hot_q * log_probas).sum(-1)  # [batch, q_shot, num_class]
        ce = F.cross_entropy(scores[0], y_q[0])
        preds_q = scores.detach().softmax(-1).argmax(2)

        return ce, preds_q
