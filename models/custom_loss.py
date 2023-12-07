import torch.nn.functional as F


def compute_loss(logits, labels, c=0.3):
    labels_thickness, labels_coordinates = labels[..., 0], labels[..., 1:]
    logits_thickness, logits_coordinates = logits[..., 0], logits[..., 1:]
    thickness_loss = F.mse_loss(logits_thickness, labels_thickness)
    coordinates_loss = F.mse_loss(logits_coordinates, labels_coordinates)
    total_loss = thickness_loss*c + coordinates_loss*(1-c)
    return total_loss
