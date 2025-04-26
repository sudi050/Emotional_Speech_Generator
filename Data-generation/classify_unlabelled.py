import os
import torch
import torchaudio
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import shutil
import torch.nn as nn
import matplotlib.pyplot as plt

# === SETTINGS ===
DATA_DIR = "unlabelled_bg_removed"  # Change as needed
IMG_SIZE = 224
BATCH_SIZE = 4
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === DATASET ===
class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform

        file_list = [f for f in os.listdir(root_dir) if f.endswith((".wav", ".mp3"))]

        # Extract labels from filenames (customize if needed)
        all_labels = set()
        for file in file_list:
            label = file.split("_")[0]
            all_labels.add(label)

        self.label_map = {label: idx for idx, label in enumerate(sorted(all_labels))}

        for file in file_list:
            filepath = os.path.join(root_dir, file)
            label = file.split("_")[0]
            self.samples.append(filepath)
            self.labels.append(self.label_map[label])

        self.spectrogram = torchaudio.transforms.MelSpectrogram()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path = self.samples[idx]
        label = self.labels[idx]

        waveform, sample_rate = torchaudio.load(audio_path)
        mel_spec = self.spectrogram(waveform).squeeze(0)

        mel_spec_normalized = (
            (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-6) * 255.0
        )
        mel_spec_normalized = mel_spec_normalized.unsqueeze(0)

        # Pad or crop to fixed size (time dimension)
        target_time_dim = 400
        current_time_dim = mel_spec_normalized.shape[2]

        if current_time_dim < target_time_dim:
            pad_amount = target_time_dim - current_time_dim
            mel_spec_normalized = torch.nn.functional.pad(
                mel_spec_normalized, (0, pad_amount)
            )
        else:
            mel_spec_normalized = mel_spec_normalized[:, :, :target_time_dim]

        mel_spec_normalized = mel_spec_normalized.squeeze(0)  # Final shape: [freq, time]

        if self.transform:
            mel_spec_np = mel_spec_normalized.numpy()
            mel_spec_transformed = self.transform(mel_spec_np)
            return mel_spec_transformed, label
        else:
            return mel_spec_normalized, label


# === TRANSFORMS ===
transform_dnn = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
)


# === MODEL ===
class SimpleDNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten [B, F, T] → [B, F*T]
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# === DATA SPLIT ===
dataset = AudioDataset(DATA_DIR, transform=transform_dnn)

# train_indices, val_indices = train_test_split(
#     list(range(len(dataset))),
#     test_size=0.2,
#     stratify=dataset.labels,
#     random_state=42
# )

# train_subset = torch.utils.data.Subset(dataset, train_indices)
# val_subset = torch.utils.data.Subset(dataset, val_indices)

# train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

# === MODEL SETUP ===
# print(dataset[0])
input_size = dataset[0][0].numel()  # Flattened spectrogram size
hidden_size = 64

CLASS_NAMES = ["anger", "disgust", "fear", "joy", "sadness"]

# === LOAD BEST MODEL ===
model_dnn = SimpleDNN(input_size, hidden_size, NUM_CLASSES).to(DEVICE)
model_dnn.load_state_dict(torch.load("emotclassifier_dnn.pth"))
model_dnn.eval()


# === SAVE HIGH CONFIDENCE PREDICTIONS ===
HIGH_PROBA_THRESHOLD = 0.95
SAVE_DIR = "high_confidence_labels"
os.makedirs(SAVE_DIR, exist_ok=True)

# Make subfolders for each class
for class_name in CLASS_NAMES:
    os.makedirs(os.path.join(SAVE_DIR, class_name), exist_ok=True)

with torch.no_grad():
    for idx in range(len(dataset)):
        mel_spec_flat, label = dataset[idx]
        input_tensor = mel_spec_flat.unsqueeze(0).to(DEVICE)
        output = model_dnn(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        prob, predicted = torch.max(probabilities, 1)

        confidence = prob.item()
        predicted_class = predicted.item()
        predicted_class_name = CLASS_NAMES[predicted_class]

        if confidence >= HIGH_PROBA_THRESHOLD:
            src_path = dataset.samples[idx]
            filename = os.path.basename(src_path)
            dst_dir = os.path.join(SAVE_DIR, predicted_class_name)
            dst_path = os.path.join(dst_dir, f"{confidence:.2f}_{filename}")
            shutil.copy(src_path, dst_path)
            print(
                f"Saved '{filename}' → class: {predicted_class_name}, confidence: {confidence:.2f}"
            )
