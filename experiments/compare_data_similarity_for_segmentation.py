import time
import gc
import os
import sys
import torch
import subprocess
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader


# Custom imports assuming they are in the parent directory
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from data_utils import ContDataset, Transform
from models import Decoder, AutoEncoder
from losses import ContrastiveLoss, DiceLoss
from metrics import pixel_wise_accuracy, evaluate_model_performance
from data_augmentation import DataAugmentation
from utils import test_visualization


# Configuration for datasets
batch_size = 64
dataset_config = {
    'image_size': (224, 224),
    'data_path': '../datasets/data',
    'flowers_data_path': '../datasets/102flowers',
    'aug_data_path': '../datasets/aug_data'
}

pre_train = {
    'num_samples': 8000,  # Size of the pre-trained dataset
    'epochs': 10,  # Total epochs for pre-training
    'learning_rate': 1e-3  # Learning rate in the pre-training phase
}

# Split configurations for fine-tuning
fine_tune_dataset_split = {
    'use_data_ratio': 0.1,  # Fine-tuning the use ratio of the dataset
    'train_ratio': 0.8,  # Fine-tuning the scale of the training dataset
    'test_ratio': 0.2,  # Fine-tuning the scale of the test dataset
}

# Training configuration for fine-tuning
fine_tune_training_config = {
    'batch_size': 64,
    'shuffle_train': True,
    'shuffle_test': False,
    'training_epochs': 20,  # Fine-tuning phase training epochs
    'learning_rate': 1e-4  # Learning rate in the pre-training phase
}

# loss object
dice_loss = DiceLoss()
contrastive_loss = ContrastiveLoss()

# # Pre-Train and Fine-tuning of pre-trained models （Use of pet-related pre-training data）


# Unzip dataset if not already present
if not os.path.exists(dataset_config['data_path']):
    subprocess.run(f'unzip ../datasets/data.zip -d {"../datasets"}',
                   shell=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
else:
    print("The data folder already exists, no need to unzip it again")

# Initialize data augmentation module
augmentor = DataAugmentation(dataset_config['data_path'],
                             dataset_config['aug_data_path'],
                             pre_train['num_samples'])
augmentor.augment_images()

# List files and count them in each directory using subprocess
data_files_count = subprocess.check_output(
    f'ls -1 {dataset_config["data_path"]} | wc -l', shell=True).strip().decode()
aug_data_files_count = subprocess.check_output(
    f'ls -1 {dataset_config["aug_data_path"]} | wc -l', shell=True).strip().decode()
print(f"Number of files in data directory: {data_files_count}")
print(
    f"Number of files in aug_data directory - Pre-training dataset: {aug_data_files_count}")

# Define a transform to convert the images to PyTorch tensors and any other desired transformations
transform = transforms.Compose([
    # Resize the image to 224x224 pixels.
    transforms.Resize(dataset_config['image_size']),
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor.
])

# Load dataset and create dataloader
dataset = ContDataset(folder_path=dataset_config['aug_data_path'],
                      folder_path1=dataset_config['data_path'],
                      transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


# Setup model and training devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = vit_b_16(pretrained=False).to(device)
decoder = Decoder(1000, 512, 3 * 224 * 224).to(device)
pre_model_related_pets = AutoEncoder(encoder, decoder).to(device)
optimizer = optim.Adam(pre_model_related_pets.parameters(),
                       lr=pre_train['learning_rate'])
mask = torch.rand(size=(1, 3, 224, 224)) > 0.5
mask = mask.to(device)
scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None


# Start the pre-training phase
print("Starting the pre-training related pets phase...")

for epoch in range(pre_train['epochs']):
    start_time = time.time()
    pre_model_related_pets.train()
    epoch_losses = []  # Collect losses for each batch to calculate epoch average

    for x, z1, z2 in dataloader:
        inputs, x1, x2 = x.to(device), z1.to(device), z2.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            try:
                p1, p2 = pre_model_related_pets(x1, mask).reshape(
                    64, 3, 224, 224), pre_model_related_pets(x2, mask).reshape(64, 3, 224, 224)
                loss = contrastive_loss(inputs, p1, p2)
                epoch_losses.append(loss.item())

            except Exception as e:
                continue  # Skip the backward pass and optimizer step if an error occurred

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

    # Calculate and print the average loss for the epoch
    epoch_avg_loss = sum(epoch_losses) / \
        len(epoch_losses) if epoch_losses else float('inf')
    end_time = time.time()
    epoch_duration = end_time - start_time
    print(
        f'Epoch {epoch+1}, Avg. Loss: {epoch_avg_loss:.3f}, Duration: {epoch_duration:.2f} seconds')

print("Pre-training related pets phase completed.")


transform = Transform(image_size=dataset_config['image_size'])
full_dataset = torchvision.datasets.OxfordIIITPet(root='../datasets',
                                                  target_types='segmentation',
                                                  transforms=transform,
                                                  download=True)


# Define the size of training and testing datasets
total_size = len(full_dataset)
used_data_size = int(total_size * fine_tune_dataset_split['use_data_ratio'])
train_size = int(used_data_size * fine_tune_dataset_split['train_ratio'])
test_size = used_data_size - train_size

# Split the dataset
indices = torch.randperm(total_size).tolist()
used_indices = indices[:used_data_size]
train_indices = used_indices[:train_size]
test_indices = used_indices[train_size:]

train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
test_dataset = torch.utils.data.Subset(full_dataset, test_indices)


print(f"Fine-tuning Original dataset size: {total_size}")
print(f"Fine-tuning Used dataset size: {used_data_size} ({fine_tune_dataset_split['use_data_ratio']*100}%)")
print(f"Fine-tuning Training dataset size: {len(train_dataset)}")
print(f"Fine-tuning Testing dataset size: {len(test_dataset)}")

train_loader = DataLoader(
    train_dataset, batch_size=fine_tune_training_config['batch_size'], shuffle=fine_tune_training_config['shuffle_train'], drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=4,
                         shuffle=fine_tune_training_config['shuffle_test'], drop_last=True)


evaluate_model_performance(pre_model_related_pets, test_loader,
                           device, mask, "the pre-trained related pets model")


# Initialize mask and model for fine-tuning
mask = torch.ones(size=(1, 3, 224, 224)).to(device)
fine_model_with_pre_related_pets = pre_model_related_pets.to(device)
optimizer = optim.Adam(fine_model_with_pre_related_pets.parameters(
), lr=fine_tune_training_config['learning_rate'])

# Start the fine-tuning process
print("Starting the fine-tuning process with the pre-trained related pets model...")

for epoch in range(fine_tune_training_config['training_epochs']):
    start_time = time.time()
    # Ensure the model is in training mode
    fine_model_with_pre_related_pets.train()
    for i, (x, y) in enumerate(train_loader):
        inputs, targets = x.to(device), y.to(device)
        optimizer.zero_grad()

        # Generate predictions using the fine-tuned model
        preds = fine_model_with_pre_related_pets(inputs, mask)
        if preds.size(0) == inputs.size(0):
            batch_size = preds.shape[0]
            preds = preds.reshape(batch_size, 3, 224, 224)

            # Compute the loss and accuracy
            loss = dice_loss(preds, targets)
            accuracy = pixel_wise_accuracy(preds, targets)

            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Print batch loss
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item():.3f}, Accuracy: {accuracy:.3f}')

    end_time = time.time()
    epoch_duration = end_time - start_time

    # Print epoch results
    print(f'Epoch {epoch+1} completed, Duration: {epoch_duration:.2f} seconds------------------------------------')

print("Fine-tuning completed.")


evaluate_model_performance(fine_model_with_pre_related_pets, test_loader, device,
                           mask, "The fine-tuned model based on pre-trained with related pets model")

test_visualization(fine_model_with_pre_related_pets, test_loader, mask, device,
                   f"The fine-tuned model based on pre-trained with related pets model", "../images/compare_data_similarity_for_segmentation")

# Freeing up graphics card memory
fine_model_with_pre_related_pets.to('cpu')
torch.cuda.empty_cache()
gc.collect()


# # Pre-Train and Fine-tuning of pre-trained models （Use of pre-training data not related to pets）


# Unzip dataset if not already present
if not os.path.exists(dataset_config['flowers_data_path']):
    subprocess.run(f'unzip ../datasets/102flowers.zip -d {"../datasets"}',
                   shell=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
else:
    print("The data folder already exists, no need to unzip it again")

# Initialize data augmentation module
augmentor = DataAugmentation(dataset_config['flowers_data_path'],
                             dataset_config['aug_data_path'],
                             pre_train['num_samples'])
augmentor.augment_images()

# List files and count them in each directory using subprocess
data_files_count = subprocess.check_output(
    f'ls -1 {dataset_config["flowers_data_path"]} | wc -l', shell=True).strip().decode()
aug_data_files_count = subprocess.check_output(
    f'ls -1 {dataset_config["aug_data_path"]} | wc -l', shell=True).strip().decode()
print(f"Number of files in data directory: {data_files_count}")
print(
    f"Number of files in aug_data directory - Pre-training dataset: {aug_data_files_count}")

# Define a transform to convert the images to PyTorch tensors and any other desired transformations
transform = transforms.Compose([
    # Resize the image to 224x224 pixels.
    transforms.Resize(dataset_config['image_size']),
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor.
])

# Load dataset and create dataloader
dataset = ContDataset(folder_path=dataset_config['aug_data_path'],
                      folder_path1=dataset_config['flowers_data_path'],
                      transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


# Setup model and training devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = vit_b_16(pretrained=False).to(device)
decoder = Decoder(1000, 512, 3 * 224 * 224).to(device)
pre_model_not_related_pets = AutoEncoder(encoder, decoder).to(device)
optimizer = optim.Adam(
    pre_model_not_related_pets.parameters(), lr=pre_train['learning_rate'])
mask = torch.rand(size=(1, 3, 224, 224)) > 0.5
mask = mask.to(device)
scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None


# Start the pre-training phase
print("Starting the pre-training not related pets phase...")

for epoch in range(pre_train['epochs']):
    start_time = time.time()
    pre_model_not_related_pets.train()
    epoch_losses = []  # Collect losses for each batch to calculate epoch average

    for x, z1, z2 in dataloader:
        inputs, x1, x2 = x.to(device), z1.to(device), z2.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            try:
                p1, p2 = pre_model_not_related_pets(x1, mask).reshape(
                    64, 3, 224, 224), pre_model_not_related_pets(x2, mask).reshape(64, 3, 224, 224)
                loss = contrastive_loss(inputs, p1, p2)
                epoch_losses.append(loss.item())

            except Exception as e:
                continue  # Skip the backward pass and optimizer step if an error occurred

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

    # Calculate and print the average loss for the epoch
    epoch_avg_loss = sum(epoch_losses) / \
        len(epoch_losses) if epoch_losses else float('inf')
    end_time = time.time()
    epoch_duration = end_time - start_time
    print(
        f'Epoch {epoch+1}, Avg. Loss: {epoch_avg_loss:.3f}, Duration: {epoch_duration:.2f} seconds')

print("Pre-training not related pets phase completed.")


evaluate_model_performance(pre_model_not_related_pets, test_loader,
                           device, mask, "the pre-trained not related pets model")


# Initialize mask and model for fine-tuning
mask = torch.ones(size=(1, 3, 224, 224)).to(device)
fine_model_with_pre_not_related_pets = pre_model_not_related_pets.to(device)
optimizer = optim.Adam(fine_model_with_pre_not_related_pets.parameters(
), lr=fine_tune_training_config['learning_rate'])

# Start the fine-tuning process
print("Starting the fine-tuning process with the pre-trained not related pets model...")

for epoch in range(fine_tune_training_config['training_epochs']):
    start_time = time.time()
    # Ensure the model is in training mode
    fine_model_with_pre_not_related_pets.train()
    for x, y in train_loader:
        inputs, targets = x.to(device), y.to(device)
        optimizer.zero_grad()

        # Generate predictions using the fine-tuned model
        preds = fine_model_with_pre_not_related_pets(inputs, mask)
        if preds.size(0) == inputs.size(0):
            batch_size = preds.shape[0]
            preds = preds.reshape(batch_size, 3, 224, 224)

            # Compute the loss and accuracy
            loss = dice_loss(preds, targets)
            accuracy = pixel_wise_accuracy(preds, targets)

            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Print batch loss
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item():.3f}, Accuracy: {accuracy:.3f}')

    end_time = time.time()
    epoch_duration = end_time - start_time

    # Print epoch results
    print(f'Epoch {epoch+1} completed, Duration: {epoch_duration:.2f} seconds------------------------------------')

print("Fine-tuning completed.")


evaluate_model_performance(fine_model_with_pre_not_related_pets, test_loader, device,
                           mask, "The fine-tuned model based on pre-trained with not related pets model")

test_visualization(fine_model_with_pre_not_related_pets, test_loader, mask, device,
                   f"The fine-tuned model based on pre-trained with not related pets model", "../images/compare_data_similarity_for_segmentation")

# Freeing up graphics card memory
fine_model_with_pre_not_related_pets.to('cpu')
torch.cuda.empty_cache()
gc.collect()
