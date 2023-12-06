from transformers import Trainer
import torch.nn.functional as F
import numpy as np


def compute_loss(self, model, inputs):
    inputs = np.array([(1, 1), (5, 1), 2])
    actual = np.array([(1, 0), (5, 1), 1.5])
    labels = inputs["labels"]
    outputs = model(inputs["image_tensors"])
    # logits = outputs[0]
    # loss = F.mse_loss(logits, labels) #change to coordinates, two end pionts and thickness

    c = 0.3
    outputs = model(coordinates1=inputs[0], coordinates2=inputs[1], **inputs)

    # labels_thickness, labels_coordinates = labels[:, 0], labels[:, 1:]
    # logits_thickness, logits_coordinates = logits[:, 0], logits[:, 1:]

    labels_thickness, labels_coordinates = actual[2], actual[:1]
    logits_thickness, logits_coordinates = inputs[2], inputs[:1]

    thickness_loss = F.mse_loss(logits_thickness, labels_thickness)
    coordinates_loss = F.mse_loss(logits_coordinates, labels_coordinates)

    total_loss = thickness_loss*c + coordinates_loss*(1-c)
    print("custom trainer")
    print(total_loss)
    return total_loss
