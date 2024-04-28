import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import SatelliteDataset
from UNet import UNet

MODEL = 'UNet'
EPOCH = 1000
LEARNING_RATE = 0.001
BATCH_SIZE = 8
IMAGE_SIZE = (400, 400)
VAL_SPILIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42
DEVICE = 'cuda'

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

##################################################
# Dataset
##################################################

noise_dataset = SatelliteDataset('Train', (IMAGE_SIZE), transform=True, add_noise=True)
clean_dataset = SatelliteDataset('Train', (IMAGE_SIZE), transform=True, add_noise=False)

train_size = int((1 - VAL_SPILIT - TEST_SPLIT) * len(noise_dataset))
val_size = int(VAL_SPILIT * len(noise_dataset))
test_size = len(noise_dataset) - train_size - val_size

train_dataset, _ = random_split(noise_dataset, [train_size, len(noise_dataset) - train_size])
_, val_dataset, test_dataset = random_split(clean_dataset, [len(clean_dataset) - val_size - test_size, val_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

##################################################
# Model
##################################################

if MODEL == 'UNet':
    def to_device(data, device):
        if isinstance(data, (list,tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)
    model = to_device(UNet(in_channels=3, out_channels=1, init_features=32), DEVICE)
elif MODEL == 'ResNet':
    pass

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

##################################################
# Training
##################################################

log_dir = f'runs/{MODEL}'
writer = SummaryWriter(log_dir=log_dir)

for epoch in range(1, EPOCH + 1):
    # train
    model.train()

    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for i, (images, labels) in progress_bar:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_description(f"Epoch {epoch}/{EPOCH}, Train Loss: {loss.item()}")

        writer.add_scalar('Loss/train', loss.item(), (epoch - 1) * len(train_dataloader) + i)

    # save
    if epoch % 10 == 0:
        save_dir = 'checkpoints'
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, f'{MODEL}_{epoch}.pth'))

    # validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataset)
    writer.add_scalar('Loss/val', avg_val_loss, epoch)

# test
model.eval()
total_test_loss = 0
with torch.no_grad():
    for images, labels in test_dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_test_loss += loss.item()
avg_test_loss = total_test_loss / len(test_dataloader)
print(f"Test Loss: {avg_test_loss}")

writer.close()