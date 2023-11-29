import torch
import cv2
import os
import numpy as np

class RasterizedDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, num_samples, transform=None): 
        self.rasterized_data = []
        self.labels = []
        for file in os.listdir(root_dir):
            if file.endswith(".png"):
                img = cv2.imread(os.path.join(root_dir, file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (32, 32))
                self.rasterized_data.append(torch.tensor(img).float())
                self.labels.append(int(file.split('.')[0]))
        self.rasterized_data = torch.stack(self.rasterized_data)
        self.labels = torch.tensor(self.labels).long()
        self.sample_idx = np.random.choice(len(self.rasterized_data), num_samples, replace=True)
        self.num_samples = num_samples
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, i):
        idx = self.sample_idx[i]
        sample = (self.rasterized_data[idx], self.labels[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample