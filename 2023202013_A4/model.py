

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import timm
from PIL import Image
import pandas as pd
from os.path import join
import numpy as np
from tqdm import tqdm

class AgeDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, annot_path, train=True):
        self.data_path = data_path
        self.annot_path = annot_path
        self.train = train

        self.ann = pd.read_csv(annot_path)
        self.files = self.ann['file_id']
        if train:
            self.ages = self.ann['age']
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        file_name = self.files[index]
        img_path = join(self.data_path, file_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        if self.train:
            age = self.ages[index]
            return image, age
        else:
            return image

    def __len__(self):
        return len(self.files)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pretrained DeiT model with distillation
model = timm.create_model('deit_base_distilled_patch16_224', pretrained=True)
num_features = model.head.in_features
model.head = nn.Linear(num_features, 1)  # Adjust this to '1' for regression

model = model.to(device)
print(device)

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Specify paths
train_path = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/train'
train_ann = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/train.csv'
train_dataset = AgeDataset(train_path, train_ann, train=True)

test_path = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/test'
test_ann = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/submission.csv'
test_dataset = AgeDataset(test_path, test_ann, train=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)  # Ensure labels are of correct shape
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader.dataset):.4f}')


num_epochs = 15
train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()  # Set the model to evaluation model
    predictions = []

    for imgs in tqdm(loader, desc="Predicting"):
        imgs = imgs.to(device)  
        preds = model(imgs).squeeze()  
        preds = preds.cpu().numpy()  # Move predictions to CPU and convert to numpy
        predictions.extend(preds)

    return predictions


preds = predict(model, test_loader, device)


submit = pd.read_csv('/kaggle/input/smai-24-age-prediction/content/faces_dataset/submission.csv')
preds = np.round(preds)  
submit['age'] = preds  


print(submit.head())
submit.to_csv('/kaggle/working/submission8.csv', index=False)


#======================================================================================================================
# 2nd Model, DeiT with data augmentation and weight decay

# DeiT base 
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import timm
from PIL import Image
import pandas as pd
from os.path import join
import numpy as np
from tqdm import tqdm

class AgeDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, annot_path, train=True):
        self.data_path = data_path
        self.annot_path = annot_path
        self.train = train
        self.ann = pd.read_csv(annot_path)
        self.files = self.ann['file_id']
        if train:
            self.ages = self.ann['age']
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),  # Random horizontal flip
                transforms.RandomRotation(15),  # Random rotation
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Random color jitter
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __getitem__(self, index):
        file_name = self.files[index]
        img_path = join(self.data_path, file_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        if self.train:
            age = self.ages[index]
            return image, age
        else:
            return image

    def __len__(self):
        return len(self.files)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pretrained DeiT model with distillation
model = timm.create_model('deit_base_distilled_patch16_224', pretrained=True)
# model = timm.create_model('deit_base_distilled_patch16_384', pretrained=True)

num_features = model.head.in_features
model.head = nn.Linear(num_features, 1)  # Adjust this to '1' for regression

model = model.to(device)
print(device)

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Specify paths
train_path = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/train'
train_ann = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/train.csv'
train_dataset = AgeDataset(train_path, train_ann, train=True)

test_path = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/test'
test_ann = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/submission.csv'
test_dataset = AgeDataset(test_path, test_ann, train=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)  # Ensure labels are of correct shape
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader.dataset):.4f}')

# Run training
num_epochs = 25
train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)

# Prediction function
@torch.no_grad()
def predict(model, loader, device):
    model.eval()  # Set the model to evaluation model
    predictions = []

    for imgs in tqdm(loader, desc="Predicting"):
        imgs = imgs.to(device)  # Ensure images are on the correct device
        preds = model(imgs).squeeze()  # Get model predictions
        preds = preds.cpu().numpy()  # Move predictions to CPU and convert to numpy
        predictions.extend(preds)

    return predictions

# Generate predictions
preds = predict(model, test_loader, device)

# Load the sample submission file
submit = pd.read_csv('/kaggle/input/smai-24-age-prediction/content/faces_dataset/submission.csv')
preds = np.round(preds)  # Round predictions if necessary for age as an integer
submit['age'] = preds  # Assign predictions

# Preview and save resultsdd
print(submit.head())
submit.to_csv('/kaggle/working/submission11.csv', index=False)

#=====================================================================================================================
# Ensemble method: Model Averaging

import pandas as pd

# Load predictions from two models
preds1 = pd.read_csv('submission13.csv')
preds2 = pd.read_csv('submission8.csv')

# Assuming the predictions are in a column named 'age'
average_preds = (preds1['age'] + preds2['age']) / 2

# Optionally, round the predictions if necessary
average_preds = average_preds.round()  # This step depends on whether you need integer values

# Save the averaged predictions to a new file
submission = preds1.copy()
submission['age'] = average_preds
submission.to_csv('averaged_output.csv', index=False)

#=====================================================================================================================
# 3rd model used: DeiT 20 epochs

import pandas as pd

# Load predictions from two models
preds1 = pd.read_csv('submission12.csv')
preds2 = pd.read_csv('averaged_output.csv')

# Assuming the predictions are in a column named 'age'
average_preds = (preds1['age'] + preds2['age']) / 2

# Optionally, round the predictions if necessary
average_preds = average_preds.round()  # This step depends on whether you need integer values

# Save the averaged predictions to a new file
submission = preds1.copy()
submission['age'] = average_preds
submission.to_csv('averaged_output4.csv', index=False)