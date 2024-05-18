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

test_dataset_path = 'C:/Users/kangm/PycharmProjects/Kaggle_competition/kaggle/Dataset/test'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        excitation = self.dropout(excitation)  # SE Block 내부에 드롭아웃 적용
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

criterion = nn.CrossEntropyLoss()
model = MyModel()
model = model.cuda()
# inputs = audio.cuda()
optimizer = AdamW(model.parameters(), lr=0.01)
c_p_w = torch.load('C:/Users/kangm/PycharmProjects/Kaggle_competition/mel_0517_last/checkpoint/127ep_ACC_0.9558.pth') ###################### 바꿔라
model.load_state_dict(c_p_w['model_state_dict'])
optimizer.load_state_dict(c_p_w['optimizer_state_dict'])

test_df = pd.read_csv('C:/Users/kangm/PycharmProjects/Kaggle_competition/kaggle/Dataset/sample_submission.csv')

test_data = []
test_file_paths = sorted(glob.glob(f"{test_dataset_path}/*.wav"))

pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))

size = 100

for curr_path in tqdm(test_file_paths):
    audio, sr = librosa.load(curr_path, res_type="kaiser_fast")

    length = audio.shape[0] / float(sr)
    mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=size)
    mels_db = librosa.power_to_db(mels, ref=np.max)
    padded = pad2d(mels_db, 100)
    resized = cv2.resize(padded, (384, 384), interpolation=cv2.INTER_LANCZOS4)

    test_data.append(resized)

X_test = np.asarray(test_data)
x_test_to_tensor = torch.from_numpy(X_test).to(torch.float32)
test_dataset = TensorDataset(x_test_to_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
test_predictions = []

with torch.no_grad():
    for inputs in tqdm(test_dataloader):
        # Expand input tensor to [N, C, H, W]
        inputs = torch.unsqueeze(inputs[0], 1)
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_predictions += predicted.detach().cpu().numpy().tolist()

test_df['label'] = test_predictions

test_df.to_csv('C:/Users/kangm/PycharmProjects/Kaggle_competition/mel_0517_last/test_result_39.csv', index=False)
