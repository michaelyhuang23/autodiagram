import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import os
import io
from PIL import Image
import pickle
from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image
import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (50000,-1))
from sklearn.model_selection import train_test_split

def read_label_file(label_file, image_id):
    with open(label_file, "rb") as openfile:
        labels = pickle.load(openfile)
    annotations = []
    for label in labels:
        annotation = {}
        annotation["image_id"] = image_id
        annotation["category_id"] = 1
        annotation["isCrowd"] = 0
        annotation["area"] = abs(label[0][0] - label[1][0]) * abs(label[0][1] - label[1][1])
        annotation["bbox"] = [*label[0], *label[1]]
        annotations.append(annotation)
    return {'image_id': image_id, 'annotations': annotations}

def read_data(root_dir):
    image_dir = os.path.join(root_dir, "imgfiles")
    label_dir = os.path.join(root_dir, "labels")
    all_annotations = []
    all_images = []
    img_seconds = {}
    for img_file in os.listdir(image_dir):
        if '.jpg' not in img_file: continue
        first = int(img_file.split('-')[0])
        second = int(img_file.split('-')[1].split('.')[0])
        if first not in img_seconds:
            img_seconds[first] = second
        else:
            img_seconds[first] = max(img_seconds[first], second)

    for label_file in sorted(os.listdir(label_dir)):
        image_id = int(label_file.split('.')[0])
        if image_id not in img_seconds: continue
        annotations = read_label_file(os.path.join(label_dir, label_file), image_id)
        all_annotations.append(annotations)
        image_path = os.path.join(image_dir, f'{image_id}-{img_seconds[image_id]}.jpg')
        all_images.append(Image.open(image_path))
    return all_images, all_annotations

images, labels = read_data('../data/detection_dataset')
images, eval_images = train_test_split(images, test_size=0.2, shuffle=False)
labels, eval_labels = train_test_split(labels, test_size=0.2, shuffle=False)

train_dataset_dict = {"images": images, "labels":labels}
eval_dataset_dict = {"images":eval_images, "labels":eval_labels}

train_dataset = Dataset.from_dict(train_dataset_dict)
eval_dataset = Dataset.from_dict(eval_dataset_dict)
dataset = DatasetDict({
    "train":train_dataset,
    "validation":eval_dataset,
})

dataset.push_to_hub("michaelyhuang23/autodiagram3")

