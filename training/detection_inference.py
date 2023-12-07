
from transformers import AutoModelForObjectDetection
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoImageProcessor
import numpy as np
import torch
from datasets import load_dataset

checkpoint = "facebook/detr-resnet-50"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
model = AutoModelForObjectDetection.from_pretrained("Alchemy5/detr-resnet-50_finetuned_cppe5")
dataset = load_dataset("michaelyhuang23/autodiagram3")


def transform_aug(data):
    images = [image.convert('RGB') for image in data['images']]
    return image_processor(images=images, return_tensors="pt")

dataset["validation"] = dataset["validation"].with_transform(transform_aug)

print(dataset["validation"][0])
output_txt = [r"""\documentclass[tikz,border=3mm]{standalone}
\begin{document}
\begin{tikzpicture}
"""]
with torch.no_grad():
    pixel_values = dataset['validation'][0]["pixel_values"][None, ...]
    pixel_mask = dataset['validation'][0]["pixel_mask"][None, ...]

    outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    target_size = pixel_values.shape[-2:]
    results = image_processor.post_process_object_detection(outputs, threshold=0.5)[0]

    for box in dataset['validation'][0]['labels']['boxes']:
        box = [round(x.item(), 2) for x in box] 
        output_txt.append(f"\draw ({box[0]*892},{box[1]*672}) -- ({box[2]*896},{box[3]*672});")

tail = r"""\end{tikzpicture}
\end{document}"""

output_txt.append(tail)

print('\n'.join(output_txt))

import cv2
from PIL import Image, ImageDraw

# convert from tensor to pil
image = Image.fromarray(np.round(255*pixel_values[0].permute(1,2,0).numpy()).astype(np.uint8))
draw = ImageDraw.Draw(image)
print(image.size)
for box in dataset['validation'][0]['labels']['boxes']:
    box = [round(x.item(), 2) for x in box] 
    box = [box[1]*800, box[0]*1066, box[3]*800, box[2]*1066]
    x1 = min(box[0], box[2])
    y1 = min(box[1], box[3])
    x2 = max(box[0], box[2])
    y2 = max(box[1], box[3])
    draw.rectangle((x1, y1, x2, y2), outline='red')

image.show()
    

    


