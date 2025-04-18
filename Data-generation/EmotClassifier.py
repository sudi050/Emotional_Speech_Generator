import os
import numpy as np
import torch
import torchaudio
from torchvision.models import resnet18
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import zipfile
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import shutil

# def unzip_file(zip_path, extract_to):
#     os.makedirs(extract_to, exist_ok=True)
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(extract_to)
#     print(f"[✅] Extracted '{zip_path}' to '{extract_to}'")

# zip_file_path = "/content/out.zip"        
# VIDEO_INPUT_DIR = "data"

unzip_file(zip_file_path, VIDEO_INPUT_DIR)
# === SETTINGS ===
DATA_DIR = "data"
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = len(os.listdir(DATA_DIR))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform

        file_list = [f for f in os.listdir(root_dir) if f.endswith(('.wav', '.mp3'))]

        # Extract labels from filenames (you can adjust this parsing as needed)
        all_labels = set()
        for file in file_list:
            label = file.split('_')[0]  # or customize label parsing
            all_labels.add(label)

        self.label_map = {label: idx for idx, label in enumerate(sorted(all_labels))}

        for file in file_list:
            filepath = os.path.join(root_dir, file)
            label = file.split('_')[0]
            self.samples.append(filepath)
            self.labels.append(self.label_map[label])

        # Define spectrogram transform
        self.spectrogram = torchaudio.transforms.MelSpectrogram()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path = self.samples[idx]
        label = self.labels[idx]

        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mel spectrogram
        mel_spec = self.spectrogram(waveform).squeeze(0)  # shape: [freq, time]

        # Normalize to 0–255 and convert to PIL image
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min()) * 255.0
        mel_spec = mel_spec.numpy().astype(np.uint8)
        mel_image = Image.fromarray(mel_spec)

        if self.transform:
            mel_image = self.transform(mel_image)

        return mel_image, label

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# === MODEL SETUP ===
model = resnet18(weights="DEFAULT")
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# === TRAINING LOOP ===
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# === SPLIT DATA ===
dataset = AudioDataset(DATA_DIR, transform=transform)

train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=dataset.labels, random_state=42)

train_subset = torch.utils.data.Subset(dataset, train_indices)
val_subset = torch.utils.data.Subset(dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

# === EARLY STOPPING CONFIG ===
early_stopping_patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0

# === TRAINING LOOP W/ VALIDATION ===
train_losses = []
val_losses = []
EPOCHS = 50
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
for epoch in range(EPOCHS):
    scheduler.step()
    model.train()
    train_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # === VALIDATION ===
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    # === EARLY STOPPING ===
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "emotclassifier1.pth")  # Save best model
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# # === LOAD BEST MODEL ===
model.load_state_dict(torch.load("emotclassifier1.pth"))
model.eval()

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='o')
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from pathlib import Path
import shutil

# === PREDICT AND SAVE HIGH CONFIDENCE FILES ===
HIGH_PROBA_THRESHOLD = 0.9
SAVE_DIR = "high_confidence"
os.makedirs(SAVE_DIR, exist_ok=True)

model.eval()

with torch.no_grad():
    for idx in range(len(dataset)):
        mel_image, label = dataset[idx]
        input_tensor = mel_image.unsqueeze(0).to(DEVICE)
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        prob, predicted = torch.max(probabilities, 1)

        confidence = prob.item()
        predicted_class = predicted.item()

        if confidence >= HIGH_PROBA_THRESHOLD:
            src_path = dataset.samples[idx]
            filename = os.path.basename(src_path)
            dst_path = os.path.join(SAVE_DIR, f"{confidence:.2f}_{filename}")
            shutil.copy(src_path, dst_path)
            print(f"Saved '{filename}' → class: {predicted_class}, confidence: {confidence:.2f}")
