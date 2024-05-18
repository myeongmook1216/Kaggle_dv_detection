import numpy as np
import pandas as pd
import glob
import os
import shutil
import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import IPython
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.models import resnet50, resnet101, resnet34
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from torchvision import models
import cv2
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

train_dataset_path = 'C:/Users/kangm/PycharmProjects/Kaggle_competition/kaggle/Dataset/train'
test_dataset_path = 'C:/Users/kangm/PycharmProjects/Kaggle_competition/kaggle/Dataset/test'


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

init_seeds(0)

pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))

import warnings
warnings.filterwarnings('ignore')
data = []
labels = []
folders = ['fake', 'real']  # fake: 0, real: 1

def stretch(sample, rate=2.0):
    stretch_sample = librosa.effects.time_stretch(sample, rate=2.0)
    return stretch_sample

def pitch(sample, sampling_rate, pitch_factor=0.8):
    pitch_sample = librosa.effects.pitch_shift(sample, sr=sampling_rate, n_steps=pitch_factor)
    return pitch_sample

size = 100

for folder in folders:
    file_paths = glob.glob(f"{train_dataset_path}/{folder}/*.wav")
    for curr_path in tqdm(file_paths):
        audio, sr = librosa.load(curr_path, res_type="kaiser_fast")  # sample rate = 22050
        length = audio.shape[0] / float(sr)

        ### 증강 2
        audio_pitched = pitch(audio, sr)
        audio_stretched = stretch(audio, rate=2.0)

        mels_1 = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=size)
        mels_2 = librosa.feature.melspectrogram(y=audio_pitched, sr=sr, n_mels=size)
        mels_3 = librosa.feature.melspectrogram(y=audio_stretched, sr=sr, n_mels=size)

        mels_db_1 = librosa.power_to_db(mels_1, ref=np.max)
        mels_db_2 = librosa.power_to_db(mels_2, ref=np.max)
        mels_db_3 = librosa.power_to_db(mels_3, ref=np.max)

        padded_1 = pad2d(mels_db_1, 100)
        padded_2 = pad2d(mels_db_2, 100)
        padded_3 = pad2d(mels_db_3, 100)

        resized_1 = cv2.resize(padded_1, (384, 384), interpolation=cv2.INTER_LANCZOS4)
        resized_2 = cv2.resize(padded_2, (384, 384), interpolation=cv2.INTER_LANCZOS4)
        resized_3 = cv2.resize(padded_3, (384, 384), interpolation=cv2.INTER_LANCZOS4)

        data.append(resized_1)
        data.append(resized_2)
        data.append(resized_3)

        labels.append(folder)
        labels.append(folder)
        labels.append(folder)


feature_df = pd.DataFrame({"features": data, "class": labels})
feature_df.head()

feature_df["class"].value_counts()

def label_encoder(column):
    le = LabelEncoder().fit(column)
    print(column.name, le.classes_)
    return le.transform(column)

feature_df["class"] = label_encoder(feature_df["class"])  # fake: 0, real: 1

X = np.asarray(feature_df["features"].tolist())
y = np.asarray(feature_df["class"].tolist())

num_labels = len(feature_df["class"].unique())

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# First step: converting to tensor
x_train_to_tensor = torch.from_numpy(X_train).to(torch.float32)
y_train_to_tensor = torch.from_numpy(y_train).to(torch.long)
x_val_to_tensor = torch.from_numpy(X_val).to(torch.float32)
y_val_to_tensor = torch.from_numpy(y_val).to(torch.long)

# Second step: Creating TensorDataset for Dataloader
train_dataset = TensorDataset(x_train_to_tensor, y_train_to_tensor)
val_dataset = TensorDataset(x_val_to_tensor, y_val_to_tensor)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, dropout_rate=0.5):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        squeeze = F.adaptive_avg_pool2d(x, 1)
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))
        excitation = excitation.view(batch_size, num_channels, 1, 1)
        scale = x * excitation
        return scale

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResidualBlock_SE(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock_SE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.se = SEBlock(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.se(out)
        out = F.relu(out)
        return out


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res1 = ResidualBlock(16, 32)
        self.res2 = ResidualBlock(32, 32)
        self.res3 = ResidualBlock(32, 64)
        self.res4 = ResidualBlock(64, 64)
        self.res5 = ResidualBlock(64, 128)
        self.res6 = ResidualBlock_SE(128, 128)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.res1(x)
        x = self.pool(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()
model = MyModel()
model = model.cuda()
optimizer = AdamW(model.parameters(), lr=0.01)

# inputs = audio.cuda()

# 모델 초기화
# size와 pad_size는 사용자가 정의한 입력 데이터의 형태에 따라 설정해야 합니다.

if not os.path.exists('C:/Users/kangm/PycharmProjects/Kaggle_competition/mel_0517_last/checkpoint'):
    os.makedirs('C:/Users/kangm/PycharmProjects/Kaggle_competition/mel_0517_last/checkpoint')

# 로그 파일 설정
log_file = 'training_log.txt'

if os.path.exists(log_file):
    os.remove(log_file)

def log_message(message):
    with open(log_file, 'a') as f:
        f.write(message + '\n')

batch_size = 32
# Model training loop
epochs = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

best_acc = 0.0

train_accuracies = []
val_accuracies = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    preds = []
    labels = []

    for inputs, label in tqdm(train_loader):
        inputs = torch.unsqueeze(inputs, 1)
        inputs = inputs.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        preds.extend(predicted.cpu().numpy())
        labels.extend(label.cpu().numpy())

    train_accuracy = accuracy_score(labels, preds)
    train_accuracies.append(train_accuracy)
    train_loss = running_loss / len(train_loader.dataset)
    log_message(f'Epoch {epoch+1}/{epochs}, Train Accuracy: {train_accuracy:.4f}, Loss: {train_loss:.4f}')
    print(f'Epoch {epoch+1}/{epochs}, Train Accuracy: {train_accuracy:.4f}, Loss: {train_loss:.4f}')

    # 검증 단계
    model.eval()
    val_running_loss = 0.0
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for inputs, label in tqdm(val_loader):
            inputs = torch.unsqueeze(inputs, 1)
            inputs = inputs.to(device)
            label = label.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, label)
            val_running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(label.cpu().numpy())

    val_accuracy = accuracy_score(val_labels, val_preds)
    val_loss = val_running_loss / len(val_loader.dataset)
    val_accuracies.append(val_accuracy)
    log_message(f'Epoch {epoch+1}/{epochs}, Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}')
    print(f'Epoch {epoch+1}/{epochs}, Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        f'checkpoint/{epoch}ep_ACC_{val_accuracy:.4f}.pth')

    if val_accuracy > best_acc:
        best_acc = val_accuracy

        shutil.copyfile(f'checkpoint/{epoch}ep_ACC_{val_accuracy:.4f}.pth',
                        f'checkpoint/best_model_{epoch}ep.pth')


plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy per Epoch')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png')
plt.show()
