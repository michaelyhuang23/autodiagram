import torch
import torch.nn.functional as F
from PIL import Image
from transformers import VisionEncoderDecoderConfig
from transformers import DonutProcessor, VisionEncoderDecoderModel, BartConfig
from datasets import load_dataset
import torch


class CustomNougatModel(nn.Module):
    def __init__(self):
        super().__init__()
        nougat_model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")
        self.encoder = nougat_model.encoder
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(602112, 1024)
        self.linear2 = nn.Linear(1024, 5)

    def forward(self, image_tensors):

        encoder_outputs = self.encoder.forward(image_tensors).last_hidden_state
        encoder_outputs = self.flatten(encoder_outputs)
        l1= self.linear(encoder_outputs)
        l2 = F.relu(l1)
        l3 = self.linear2(l2)
        return l3


