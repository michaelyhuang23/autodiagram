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
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CustomNougatModel()
dataset = ClassifyDataset(root_dir='../data/classification_dataset')
config = AutoConfig.from_pretrained("facebook/nougat-base")
print(config)

train_dataset, val_dataset = train_test_split(dataset, test_size=0.2)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4)
val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=4)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.to(device)
model.train()

for epoch in range(100):
    torch.save(model.state_dict(), 'result/initial.pth')
    print("Epoch:", epoch)
    for idx, batch in enumerate(train_dataloader):
        pixel_values, labels = batch
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)
        outputs = model(pixel_values)
        loss = compute_loss(outputs, labels)
        print(f"Epoch: {epoch}, Loss:", loss.item(), f"{idx+1}/{len(train_dataloader)}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.save(model.state_dict(), f'result/epoch{epoch}.pth')
