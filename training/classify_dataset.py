import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import os
import io
from PIL import Image
import pickle

class ClassifyDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.read_data(root_dir)

    def read_data(self, root_dir):
        self.labels = []
        self.images = []
        img_dir = os.path.join(root_dir, "imgfiles")
        labels_dir = os.path.join(root_dir, "labels")
        img_seconds = {}
        for img_file in os.listdir(img_dir):
            if '.jpg' not in img_file: continue
            first = int(img_file.split('-')[0])
            second = int(img_file.split('-')[1].split('.')[0])
            if first not in img_seconds:
                img_seconds[first] = second
            else:
                img_seconds[first] = max(img_seconds[first], second)
        label_ids = set([int(label_file.split('.')[0]) for label_file in os.listdir(labels_dir) if '.pkl' in label_file])
        label_ids = label_ids.intersection(img_seconds.keys())

        for label_id in label_ids:
            label_path = os.path.join(labels_dir, f'{label_id}.pkl')
            with open(label_path, "rb") as openfile:
                labels = pickle.load(openfile)
            labels = [torch.tensor([label[2], *label[0], *label[1]]).float() for label in labels]
            if img_seconds[label_id]+1 != len(labels): continue
            self.labels += labels
            for j in range(img_seconds[label_id]+1):
                file_path = os.path.join(img_dir, f"{label_id}-{j}.jpg")
                image = Image.open(file_path)
                self.images.append(torchvision.transforms.functional.to_tensor(image).float())

    def read_labels(self, label_folder):
        for label_file in os.listdir(label_folder):
            if '.pkl' not in label_file: continue
            label_path = os.path.join(label_folder, label_file)
            with open(label_path, "rb") as openfile:
                labels = pickle.load(openfile)
            labels = [torch.tensor([label[2], *label[0], *label[1]]) for label in labels]
            
        return torch.stack(labels)

    def parse_labels(self, label_file):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        all_labels = []
        self.images_lens = []
        for line in lines:
            labels = [data_point for data_point in line.replace('[',']').split(']') if len(data_point) > 0 and data_point != '\n']
            labels = [eval(label) for label in labels]
            self.images_lens.append(len(labels))
            labels = [torch.tensor([label[2], *label[0], *label[1]]) for label in labels]
            all_labels += labels
        return torch.stack(all_labels)

    def load_images(self, img_dir):
        images = []
        # Iterate through all files in the folder
        for i, image_len in enumerate(self.images_lens):
            for j in range(image_len):
                file_path = os.path.join(img_dir, f"line_drawing_test_{i}-{j}.jpg")
                image = Image.open(file_path)
                # Append the resized image and filename to the dataset
                images.append(transforms.functional.to_tensor(image).float())
        return images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[int(index)]
        label = self.labels[int(index)]
        return image, label
