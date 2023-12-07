from transformers import AutoModelForObjectDetection
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoImageProcessor
import numpy as np
import torch
from datasets import load_dataset

checkpoint = "facebook/detr-resnet-50"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
dataset = load_dataset("michaelyhuang23/autodiagram3")

images, annotation = dataset['train']['images'], dataset['train']['labels']

def transform_aug(data):
    images = [image.convert('RGB') for image in data['images']]
    return image_processor(images=images, annotations=data['labels'], return_tensors="pt")

dataset["train"] = dataset["train"].with_transform(transform_aug)
#image = dataset["train"][0]["image"]
#annotations = dataset["train"][0]["objects"]
# categories = dataset["train"].features["objects"].feature["category"].names
# id2label = {index: x for index, x in enumerate(categories, start=0)}
# label2id = {v: k for k, v in id2label.items()}
model = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    ignore_mismatched_sizes=True,
)

training_args = TrainingArguments(
    output_dir="detr-resnet-50_finetuned_cppe5",
    per_device_train_batch_size=8,
    num_train_epochs=10,
    fp16=True,
    save_steps=200,
    logging_steps=50,
    learning_rate=1e-5,
    weight_decay=1e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=True,
)

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=dataset["train"],
    tokenizer=image_processor,
)
trainer.train()