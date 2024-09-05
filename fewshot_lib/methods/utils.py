import torch
from torch import Tensor as tensor
import torch.nn as nn


def get_one_hot(y_s: torch.tensor, num_classes: int):
    """
        args:
            y_s : torch.Tensor of shape [n_task, shot]
        returns
            y_s : torch.Tensor of shape [n_task, shot, num_classes]
    """
    one_hot_size = list(y_s.size()) + [num_classes]
    one_hot = torch.zeros(one_hot_size, device=y_s.device)
    one_hot.scatter_(-1, y_s.unsqueeze(-1), 1)
    return one_hot


def compute_centroids(z_s: torch.tensor,
                      y_s: torch.tensor):
    """
    inputs:
        z_s : torch.Tensor of size [batch_size, s_shot, d]
        y_s : torch.Tensor of size [batch_size, s_shot]

    updates :
        centroids : torch.Tensor of size [n_task, num_class, d]
    """
    one_hot = get_one_hot(y_s, num_classes=y_s.unique().size(0)).transpose(1, 2)  # [batch, K, s_shot]
    centroids = one_hot.bmm(z_s) / one_hot.sum(-1, keepdim=True)  # [batch, K, d]
    return centroids


def extract_features(x: tensor, model: nn.Module):
    """
    Extract features from support and query set using the provided model
        args:
            x_s : torch.Tensor of size [batch, s_shot, c, h, w]
        returns
            z_s : torch.Tensor of shape [batch, s_shot, d]
            z_s : torch.Tensor of shape [batch, q_shot, d]
    """
    batch, shot = x.size()[:2]
    feat_dim = x.size()[-3:]
    z, _ = model(x.view(batch * shot, *feat_dim), is_feat=True)
    z = z.view(batch, shot, -1)  # [batch, s_shot, d]
    return z

def l2_distance_to_prototypes(samples: tensor, prototypes: tensor):
    """
    Compute prediction logits from their euclidean distance to support set prototypes.
    Args:
        samples: features of the items to classify of shape (n_samples, feature_dimension)
    Returns:
        prediction logits of shape (n_samples, n_classes)
    """
    return -torch.cdist(samples, prototypes)

def cosine_distance_to_prototypes(samples: tensor, prototypes: tensor):
    """
    Compute prediction logits from their cosine distance to support set prototypes.
    Args:
        samples: features of the items to classify of shape (n_samples, feature_dimension)
    Returns:
        prediction logits of shape (n_samples, n_classes)
    """
    return (
        nn.functional.normalize(samples, dim=1)
        @ nn.functional.normalize(prototypes, dim=1).T
    )