import numpy as np
import pandas as pd
import torch
import math
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.nn.functional as F1
from torchvision.io import read_image
from torchvision.transforms import Resize, ToTensor
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset class
data = pd.read_csv('train_val_set.csv')
train_df, val_df = train_test_split(data, test_size=0.2, shuffle=True, random_state=42)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        anchor_image = read_image(os.path.join('all_dynamic_hum_crop', self.dataframe.iloc[idx]['anchor']))
        melody_image = read_image(os.path.join('all_dynamic_melody_crop', self.dataframe.iloc[idx]['melody']))
        label = self.dataframe.iloc[idx]['label']
        
        anchor_image = F.resize(anchor_image, (600, 800), antialias=True)/255.0
        melody_image = F.resize(melody_image, (600, 800), antialias=True)/255.0

        return anchor_image, melody_image, label


# Create datasets and dataloaders
train_dataset = CustomDataset(train_df)
val_dataset = CustomDataset(val_df)

# Define model
class TowerModel(nn.Module):
    def __init__(self):
        super(TowerModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        return self.conv_layers(x)

model = TowerModel().to(device)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)

#model.load_state_dict(torch.load('checkpoints3/model_epoch_40.pth'))

def forward_loss(embedding1, embedding2, label):
    d = F1.cosine_similarity(embedding1, embedding2, dim=1)
    #print(d)
    #print(d.shape)
    loss = torch.mean((1-label) * torch.pow(1-d, 2) + label * torch.pow(torch.clamp(d, min=0.0), 2))
    #print(loss)
    return loss

optimizer = optim.Adam(model.parameters(), lr=1e-5)

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
print('Data read successfully')

print('Model Training started')
train_loss = []
validation_loss = []
num_epochs = 40
for epoch in range(0, 40):
    # Training
    model.train()
    total_loss = 0
    for anchor_images, melody_images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        anchor_images, melody_images, labels = anchor_images.to(device), melody_images.to(device), labels.to(device)
        optimizer.zero_grad()
        anchor_outputs = model(anchor_images)
        melody_outputs = model(melody_images)
        loss = forward_loss(anchor_outputs, melody_outputs, labels)
        #print(loss)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()   
    
    avg_loss = total_loss/ (math.ceil(len(train_df) / batch_size))
    train_loss.append(avg_loss)
    

    #Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for anchor_images, melody_images, labels in tqdm(val_loader, desc=f"Validation {epoch+1}/{num_epochs}"):
            anchor_images, melody_images, labels = anchor_images.to(device), melody_images.to(device), labels.to(device)
            
            anchor_outputs = model(anchor_images)
            melody_outputs = model(melody_images)

            val_loss = forward_loss(anchor_outputs, melody_outputs, labels)
            total_val_loss += val_loss.item()
            
    avg_val_loss = total_val_loss/(math.ceil(len(val_df) / batch_size))
    validation_loss.append(avg_val_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.5f}, Validation Loss: {avg_val_loss:.5f}')

    torch.save(model.state_dict(), f'checkpoints3/model_epoch_{epoch+1}.pth')


