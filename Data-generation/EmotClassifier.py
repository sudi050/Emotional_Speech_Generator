import os
import numpy as np
import torch
import torchaudio
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import zipfile

# def unzip_file(zip_path, extract_to):
#     os.makedirs(extract_to, exist_ok=True)
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(extract_to)
#     print(f"[✅] Extracted '{zip_path}' to '{extract_to}'")

# zip_file_path = "/content/out.zip"        # Replace with your zip file name
# VIDEO_INPUT_DIR = "data"

# unzip_file(zip_file_path, VIDEO_INPUT_DIR)
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
            label = file.split('.')[0]  # or customize label parsing
            all_labels.add(label)

        self.label_map = {label: idx for idx, label in enumerate(sorted(all_labels))}

        for file in file_list:
            filepath = os.path.join(root_dir, file)
            label = file.split('.')[0]
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

EPOCHS = 50
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
