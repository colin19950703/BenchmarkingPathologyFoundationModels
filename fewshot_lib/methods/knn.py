import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from fewshot_lib.methods.utils import extract_features
from fewshot_lib.methods.method import FSmethod


class KNN(FSmethod):
    '''
    KNN method
    '''

    def __init__(self, opt: argparse.Namespace, normalize=True):
        super().__init__(opt)
        self.normalize = normalize
        self.k = opt.num_support
    def forward(self,
                x_s: torch.tensor,
                x_q: torch.tensor,
                y_s: torch.tensor,
                y_q: torch.tensor,
                model: nn.Module):
        """
        inputs:
            x_s : torch.Tensor of shape [s_shot, c, h, w]
            x_q : torch.Tensor of shape [q_shot, c, h, w]
            y_s : torch.Tensor of shape [s_shot]
            y_q : torch.Tensor of shape [q_shot]
        """
        num_classes = y_s.unique().size(0)
        if not self.training:
            with torch.no_grad():
                z_s = extract_features(x_s, model)
                z_q = extract_features(x_q, model)
        else:
            z_s = extract_features(x_s, model)
            z_q = extract_features(x_q, model)

        if self.normalize:
            z_s = F.normalize(z_s, dim=2)
            z_q = F.normalize(z_q, dim=2)

        # Flatten the feature maps for distance calculation
        z_s = z_s.view(z_s.shape[0], z_s.shape[1], -1)
        z_q = z_q.view(z_q.shape[0], z_q.shape[1], -1)

        # Calculate pairwise distances between support and query features
        dists = torch.cdist(z_q[0], z_s[0], p=2)  # Euclidean distance

        # Get the indices that would sort each distance tensor
        sorted_indices = torch.argsort(dists, dim=-1, descending=False)  # [60, way*shot]
        # Select the k nearest neighbors
        knn_indices = sorted_indices[:, :self.k]  # [60, k]
        knn_labels = y_s[0][knn_indices]  # [60, k]

        # Majority vote for the predicted labels
        preds_q = torch.mode(knn_labels, dim=-1).values
        preds_q = preds_q.unsqueeze(0)
        ce = torch.zeros(1)

        return ce, preds_q
