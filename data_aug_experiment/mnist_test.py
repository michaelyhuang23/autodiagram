import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import cv2
import numpy as np
from vector_dataset import RasterizedDataset
from augmentations import vert_curve_image_raw, hori_curve_image_raw

def display_img(img):
    img = img.permute(1, 2, 0)  # Change from CxHxW to HxWxC
    img = img.numpy()  # Convert to a NumPy array
    img = (img * 255).astype('uint8')  # Rescale to [0, 255]
    # Display the image using OpenCV
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# transforms.RandomResizedCrop((32,32)), 
augmentations = [transforms.Lambda(lambda x : vert_curve_image_raw(x, 0.05)),
                 transforms.Lambda(lambda x : hori_curve_image_raw(x, 0.05)),
                 transforms.RandomRotation(10)]

test_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32,32))])

train_transform = transforms.Compose([transforms.Normalize((0), (255)), transforms.RandomApply(augmentations, p=1)])


trainset = RasterizedDataset('data/train', 100000, train_transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=test_transform)

testloader = DataLoader(testset, batch_size=64, shuffle=False)

# display_img(trainset[2][0])
# display_img(testset[2][0])



class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 1000)
        self.fc2 = nn.Linear(1000, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = VanillaCNN()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.003)

running_loss = 0.0
print(len(trainloader))
for i, data in enumerate(trainloader):
    inputs, labels = data

    optimizer.zero_grad()

    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if i % 20 == 19:    # print every 100 mini-batches
        print(f"{i} loss: {running_loss / 100:.3f}")
        running_loss = 0.0

        total, correct = 0, 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'{i} acc: ', correct/total)


