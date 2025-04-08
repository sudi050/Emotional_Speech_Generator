import os
import numpy as np
import torch
import torchaudio
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

# === SETTINGS ===
DATA_DIR = "data"
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = len(os.listdir(DATA_DIR))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === AUDIO TO SPECTROGRAM IMAGE DATASET ===
class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        self.label_map = {cls: idx for idx, cls in enumerate(sorted(os.listdir(root_dir)))}

        for cls in self.label_map:
            cls_folder = os.path.join(root_dir, cls)
            for file in os.listdir(cls_folder):
                if file.endswith(('.wav', '.mp3')):
                    self.samples.append(os.path.join(cls_folder, file))
                    self.labels.append(self.label_map[cls])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path = self.samples[idx]
        label = self.labels[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to Mel Spectrogram
        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate)(waveform)
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)

        # Convert to PIL Image
        mel_image = mel_spec_db.squeeze().numpy()
        mel_image = Image.fromarray(mel_image).convert("L")  # grayscale

        if self.transform:
            mel_image = self.transform(mel_image)

        return mel_image, label

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel for VGG
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === DATA LOADING ===
dataset = AudioDataset(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === VGG16 MODEL SETUP ===
model = models.vgg16(pretrained=True)
model.classifier[6] = torch.nn.Linear(4096, NUM_CLASSES)
model = model.to(DEVICE)

# === TRAINING LOOP ===
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for images, labels in tqdm(dataloader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss:.4f}")
