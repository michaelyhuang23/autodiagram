from datasets import load_dataset
from transformers import VisionEncoderDecoderConfig
from transformers import DonutProcessor, VisionEncoderDecoderModel, BartConfig
import json
import random
from typing import Any, List, Tuple
import torch
from datasets import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path
import re
from nltk import edit_distance
import numpy as np
import math
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

dataset = load_dataset("Alchemy5/autodiagram")
max_length = 128
image_size = [1280, 960]
config = VisionEncoderDecoderConfig.from_pretrained("facebook/nougat-base")
config.encoder.image_size = image_size
config.decoder.max_length = max_length

processor = DonutProcessor.from_pretrained("facebook/nougat-base")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base", config=config)

processor.feature_extractor.size = image_size[::-1] # should be (width, height)
processor.feature_extractor.do_align_long_axis = False
dataset = load_dataset("Alchemy5/autodiagram")
train_dataset = dataset["train"]
val_dataset = dataset["validation"]

transform = transforms.ToTensor()
image, labels = train_dataset[0]["images"], train_dataset[0]["tex"]
pixel_values = transform(image)
labels = torch.tensor(processor.tokenizer(labels)["input_ids"])

def format_dataset(split):
    images = []
    labels = []
    for sample in dataset[split]:
        image = transform(sample["images"])
        label = torch.tensor(processor.tokenizer(sample["tex"], padding='max_length', truncation=True, max_length = 600)["input_ids"])
        decoder_input = torch.tensor(processor.tokenizer("Convert this image into tex:", padding='max_length', truncation=True, max_length = 600)["input_ids"])
        images.append(image)
        labels.append(label)
    dataset_new = [(image, decoder_input, label) for item in zip(images, labels)]
    return dataset_new

def format_eval_dataset(split):
    images = []
    labels = []
    for sample in dataset[split]:
        image = transform(sample["images"])
        label = sample["tex"]
        decoder_input = torch.tensor(processor.tokenizer("Convert this image into tex:", padding='max_length', truncation=True, max_length = 600)["input_ids"])
        images.append(image)
        labels.append(label)
    dataset_new = [(image, decoder_input, label) for item in zip(images, labels)]
    return dataset_new

train_dataset = format_dataset("train")
val_dataset = format_eval_dataset("validation")

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

class DonutModelPLModule(pl.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        pixel_values, decoder_input_ids, labels = batch
        outputs = self.model(pixel_values,
                             decoder_input_ids=decoder_input_ids[:, :-1],
                             labels=labels[:, 1:])
        loss = outputs.loss
        self.log_dict({"train_loss": loss}, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        pixel_values, decoder_input_ids, answers = batch

        outputs = self.model.generate(pixel_values,
                                   decoder_input_ids=decoder_input_ids,
                                   max_length=600,
                                   early_stopping=True,
                                   pad_token_id=self.processor.tokenizer.pad_token_id,
                                   eos_token_id=self.processor.tokenizer.eos_token_id,
                                   use_cache=True,
                                   num_beams=1,
                                   bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                                   return_dict_in_generate=True,)

        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            #seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            predictions.append(seq)

        scores = list()
        for pred, answer in zip(predictions, answers):
            #pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            #answer = re.sub(r"<.*?>", "", answer, count=1)
            answer = answer.replace(self.processor.tokenizer.eos_token, "")
            score = edit_distance(pred, answer) / max(len(pred), len(answer))
            scores.append(score)
            
            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")
        self.validation_step_outputs.append(scores)
        return scores

    def on_validation_epoch_end(self):
        num_of_loaders = 1
        if num_of_loaders == 1:
            validation_step_outputs = [self.validation_step_outputs]
        assert len(validation_step_outputs) == num_of_loaders
        cnt = [0] * num_of_loaders
        total_metric = [0] * num_of_loaders
        val_metric = [0] * num_of_loaders
        for i, results in enumerate(validation_step_outputs):
            for scores in results:
                cnt[i] += len(scores)
                total_metric[i] += np.sum(scores)
            val_metric[i] = total_metric[i] / cnt[i]
            val_metric_name = f"val_metric_{i}th_dataset"
            self.log_dict({val_metric_name: val_metric[i]}, sync_dist=True)
        self.log_dict({"val_metric": np.sum(total_metric) / np.sum(cnt)}, sync_dist=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr"))
        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader
    

config = {"max_epochs":5,
          "val_check_interval":0.2,
          "check_val_every_n_epoch":1,
          "gradient_clip_val":1.0,
          "num_training_samples_per_epoch": 800,
          "lr":3e-5,
          "train_batch_sizes": [1],
          "val_batch_sizes": [1],
          "num_nodes": 1,
          "warmup_steps": 300,
          "result_path": "result", # ensure dir named result
          "verbose": True,
          }

model_module = DonutModelPLModule(config, processor, model)

trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config.get("max_epochs"),
        val_check_interval=config.get("val_check_interval"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision=16,
        num_sanity_val_steps=0,
        #logger=wandb_logger,
        #callbacks=[lr_callback, checkpoint_callback],
)

trainer.fit(model_module)