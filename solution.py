import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import SatelliteDataset
from UNet import UNet

MODEL = 'UNet'
EPOCH = 1000
LEARNING_RATE = 0.0001
BATCH_SIZE = 8
IMAGE_SIZE = (400, 400)
VAL_SPILIT = 0.1
TEST_SPLIT = 0.1
DEVICE = 'cuda'

##################################################
# Dataset
##################################################

noise_dataset = SatelliteDataset('Train', (IMAGE_SIZE), transform=True, add_noise=True)
clean_dataset = SatelliteDataset('Train', (IMAGE_SIZE), transform=True, add_noise=False)

indices = list(range(len(noise_dataset)))
random.shuffle(indices)

train_size = int((1 - VAL_SPILIT - TEST_SPLIT) * len(noise_dataset))
val_size = int(VAL_SPILIT * len(noise_dataset))
test_size = len(noise_dataset) - train_size - val_size

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size+val_size]
test_indices = indices[train_size+val_size:]

train_dataset = Subset(noise_dataset, train_indices)
val_dataset = Subset(clean_dataset, val_indices)
test_dataset = Subset(clean_dataset, test_indices)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

##################################################
# Model
##################################################

if MODEL == 'UNet':
    model = UNet(in_channels=3, out_channels=1, init_features=32).to(DEVICE)
elif MODEL == 'resUNet':
    model = resUNet(in_channels=3, out_channels=1, init_features=32).to(DEVICE)
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
    total_train_loss = 0.0
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for i, (images, labels) in progress_bar:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)
        total_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        # print('loss: ', loss.item())
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} grad norm: {param.grad.norm()}")
        optimizer.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    writer.add_scalar('Loss/train', avg_train_loss, epoch)

    params = []
    for param in model.parameters():
        params += list(param.detach().cpu().numpy().flatten())
    print(f"Epoch {epoch}/{EPOCH}, Mean Params: {np.mean(params):.4f}, Max Params: {np.max(params):.4f}, Min Params: {np.min(params):.4f}")

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

    avg_val_loss = total_val_loss / len(val_dataloader)
    writer.add_scalar('Loss/val', avg_val_loss, epoch)

    print(f"Epoch {epoch}/{EPOCH}, Avg Val Loss: {avg_val_loss:.4f}, Avg Train Loss: {avg_train_loss:.4f}")

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