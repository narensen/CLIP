import torch
import torch.nn as nn
import torch.nn.functional as F


def clip_loss(image_features, text_features, logit_scale):

    text_features = F.normalize(text_features, dim=-1)
    image_features = F.normalize(image_features, dim=-1)

    logits = logit_scale * image_features @ text_features.T
    labels = torch.arange(image_features.shape[0], device=image_features.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2

    