import os
import random
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

class RandomFlipPair(object):
    def __call__(self, image, label):
        flip_type = random.choice(['hflip', 'vflip', 'hvflip', 'none'])
        if flip_type == 'hflip':
            return F.hflip(image), F.hflip(label)
        elif flip_type == 'vflip':
            return F.vflip(image), F.vflip(label)
        elif flip_type == 'hvflip':
            return F.hflip(F.vflip(image)), F.hflip(F.vflip(label))
        return image, label

class SatelliteDataset(Dataset):
    def __init__(self, root_dir, image_size, transform=False, add_noise=False):
        self.root_dir = root_dir
        self.transform = transform
        self.add_noise = add_noise
        self.images_dir = os.path.join(root_dir, 'images')
        self.labels_dir = os.path.join(root_dir, 'labels')
        self.images = os.listdir(self.images_dir)
        self.img_transform = Compose([
            Resize(image_size),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.label_transform = Compose([
            Resize(image_size),
            ToTensor()
        ])
        self.flip = RandomFlipPair()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.root_dir == 'Train':
            mask_name = img_name.replace('sat.jpg', 'mask.png')
            label_path = os.path.join(self.labels_dir, mask_name)
            label = Image.open(label_path).convert('L')

            if self.transform:
                image, label = self.flip(image, label)

            image = self.img_transform(image)
            label = self.label_transform(label)

            if self.add_noise and random.random() < 0.8:
                noise_type = random.choice(['gaussian', 'salt_pepper', 'stripe'])
                if noise_type == 'gaussian':
                    image = torch.tensor(np.array(image) + np.random.normal(0, 0.5, image.size()), dtype=torch.float)
                elif noise_type == 'salt_pepper':
                    image = torch.tensor(np.array(image) + np.random.randint(0, 256, image.size()) * (np.random.rand(*image.size()) < 0.1), dtype=torch.float)
                elif noise_type == 'stripe':
                    img_arr = np.array(image)
                    img_height, img_width, _ = img_arr.shape
                    num_stripes = random.randint(5, 10)
                    for _ in range(num_stripes):
                        img_arr = np.transpose(img_arr, (1, 2, 0))
                        stripe_start = random.randint(0, img_width - 1)
                        stripe_color = np.random.randint(0, 256, size=3) / 255
                        img_arr[:, stripe_start, :] = stripe_color
                        img_arr = np.transpose(img_arr, (2, 0, 1))

                    image = torch.tensor(img_arr, dtype=torch.float)

            return image, label

        else:
            image = self.img_transform(image)
            return image