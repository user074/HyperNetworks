import torch
import torchvision.transforms as transforms
from torchvision.datasets.coco import CocoDetection
from torch.utils.data import DataLoader

from torch.autograd import Variable
import torch.nn as nn
from pycocotools.coco import COCO

import argparse

import torch.optim as optim

from segmentation_net import SegmentationNetwork

# ... [other imports and initial setups]

########### Data Loader ###############
import torchvision.datasets as datasets

# ... [your other imports]

# Data augmentation and normalization for training
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]),
}

data_dir = '../data/VOCdevkit/VOC2012/'  # adjust if necessary
image_datasets = {x: datasets.VOCSegmentation(data_dir, year='2012', image_set=x, download=True, transform=data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# ... [other setups]

############

net = SegmentationNetwork(num_classes=92)  # COCO has 91 stuff categories + 1 background
# ... [other initializations]

# Use CrossEntropyLoss for pixel-wise comparison
criterion = nn.CrossEntropyLoss()

# ... [training loop]

for i, data in enumerate(trainloader, 0):
    inputs, annotations = data
    masks = # Convert annotations to masks. This will require additional code.

    inputs, masks = Variable(inputs.cuda()), Variable(masks.cuda())

    # ... [rest of the loop]

    outputs = net(inputs)
    loss = criterion(outputs, masks)
    # ... [rest of the training loop]

    # Compute segmentation metric (e.g., IoU) 
    # ...

print('Finished Training')
