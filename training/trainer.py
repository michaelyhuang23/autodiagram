import datasets
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel
from torch.utils.data import DataLoader
import torch
from transformers import Trainer
from transformers import VisionEncoderDecoderConfig
import pytorch_lightning as pl
from torchvision import transforms
from transformers import GPT2TokenizerFast

config = VisionEncoderDecoderConfig.from_pretrained("facebook/nougat-base")
processor = DonutProcessor.from_pretrained("facebook/nougat-base")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base", config=config)
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s_cord-v2>'])[0]
dataset = load_dataset("michaelyhuang23/autodiagram", cache_dir="/n/holystore01/LABS/iaifi_lab/Users/vhariprasad/aim_datasets")

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        pixel_values = self.processor(item["images"], random_padding=True, return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze()
        input_ids = processor.tokenizer(item["tex"], max_length = 512, padding="max_length", truncation=True, return_tensors="pt")["input_ids"].squeeze(0)
        return {"pixel_values":pixel_values, "labels":input_ids}

train_dataset = ImageCaptioningDataset(dataset["train"], processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4)
eval_dataset = ImageCaptioningDataset(dataset["validation"], processor)
eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=4)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.train()

for epoch in range(5):
    torch.save(model.state_dict(), 'result/initial.pth')
    print("Epoch:", epoch)
    for idx, batch in enumerate(train_dataloader):
        input_ids = batch.pop("labels").to(device)
        pixel_values = batch.pop("pixel_values").to(device)
        outputs = model(pixel_values=pixel_values,labels=input_ids)
        loss = outputs.loss
        print(f"Epoch: {epoch}, Loss:", loss.item(), f"{idx+1}/{len(train_dataloader)}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.save(model.state_dict(), f'result/epoch{epoch}.pth')
