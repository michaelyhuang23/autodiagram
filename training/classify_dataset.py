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
        self.source_images_num = 0
        self.labels = self.read_labels(os.path.join(root_dir, "line_coords.pickle"))
        self.images = self.load_images(os.path.join(root_dir, "imgfiles"))

    def read_labels(self, label_file):
        labels = []
        with open(label_file, "rb") as openfile:
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
        print("index:", index)
        image = self.images[int(index)]
        label = self.labels[int(index)]
        return image, label

dataset = ClassifyDataset(root_dir='../data/classification_dataset')