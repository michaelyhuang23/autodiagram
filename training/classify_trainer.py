import datasets
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel
from torch.utils.data import DataLoader
import torch
from transformers import Trainer
from transformers import VisionEncoderDecoderConfig
from torchvision import transforms
from transformers import GPT2TokenizerFast
import sys
sys.path.append('..')
from models.custom_loss import compute_loss
from models.nougat_classify import CustomNougatModel
from training.classify_dataset import ClassifyDataset
from transformers import AutoConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CustomNougatModel()
dataset = ClassifyDataset(root_dir='../data/detection_dataset')
#config = AutoConfig.from_pretrained("facebook/nougat-base")
print('done loading dataset')

data_loader = DataLoader(dataset, batch_size=4, shuffle=True)


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.to(device)
model.train()

print("Training...")

for epoch in range(100):
    torch.save(model.state_dict(), 'result/initial.pth')
    print("Epoch:", epoch)
    for idx, batch in enumerate(data_loader):
        pixel_values, labels = batch
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)
        outputs = model(pixel_values)
        loss = compute_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch: {epoch}, Loss:", loss.cpu().item(), f"{idx+1}/{len(data_loader)}")
        if idx % 100 == 0:
            torch.save(model.state_dict(), f'result/epoch{epoch}-{idx}.pth')
    torch.save(model.state_dict(), f'result/epoch{epoch}.pth')
