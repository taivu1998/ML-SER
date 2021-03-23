import os
import sys
import time
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.fftpack import fft 
from scipy import signal
from scipy.io import wavfile
import librosa
import sklearn
from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


METADATA_DIR = './datasets/data_ravdess_savee.csv'
METADATA_TRAIN_PATH = './datasets/data_ravdess_savee_train.csv'
METADATA_VALID_PATH = './datasets/data_ravdess_savee_valid.csv'
METADATA_TEST_PATH = './datasets/data_ravdess_savee_test.csv'


INDICES_TO_EMOTIONS = {
    0: 'neutral',
    1: 'calm',
    2: 'happy',
    3: 'sad',
    4: 'angry',
    5: 'fearful',
    6: 'disgust', 
    7: 'surprised',
}
EMOTIONS_TO_INDICES = {value:key for key, value in INDICES_TO_EMOTIONS.items()}
INDICES_TO_GENDERS = ['female', 'male']
GENDERS_TO_INDICES = dict(zip(INDICES_TO_GENDERS, range(len(INDICES_TO_GENDERS))))


config = {
    "features": "lms",
    "normalize": False,
    "sr": 16000,
    "n_mels": 128, 
    "n_fft":512,
    "hop_length": 128,
    "win_length": 512,
    "wav_max_length": 512, 
    "n_mfcc": 20,
    "augment": "all",
}


def load_waveform(wav_path, sr=22050, library="librosa"):
    wav, sr = librosa.load(wav_path, sr)
    wav, _ = librosa.effects.trim(wav)
    return wav


def extract_mfcc_features(wav, config=None, library="librosa"):
    if library == "librosa":
        features = librosa.feature.mfcc(wav, sr=config["sr"], 
                                        n_mfcc=config["n_mfcc"],
                                        n_mels=config["n_mels"], 
                                        n_fft=config["n_fft"], 
                                        hop_length=config["hop_length"], 
                                        win_length=config["win_length"])
        features = features.T
        if config["normalize"]:
            features = librosa.util.normalize(features)
        return features
    
    else:
        return None


def extract_log_melspectrogram_features(wav, config=None, library="librosa"):
    if library == "librosa":
        features = librosa.feature.melspectrogram(wav, sr=config["sr"], 
                                                  n_mels=config["n_mels"], 
                                                  n_fft=config["n_fft"], 
                                                  hop_length=config["hop_length"], 
                                                  win_length=config["win_length"])
        eps = 1e-10
        features = np.log(features.T + eps)
#         features = librosa.power_to_db(features, ref=np.max)
        if config["normalize"]:
            features = librosa.util.normalize(features)
        return features
    
    else:
        return None


def extract_features(wav, feature_type="mfcc", config=None, reduction=None, library="librosa"):
    if feature_type == "mfcc":
        features = extract_mfcc_features(wav, config=config, library=library)
        if reduction == "mean":
            features = np.mean(features, axis=0)
        return features
    
    elif feature_type == "lms":
        features = extract_log_melspectrogram_features(wav, config=config, library=library)
        if reduction == "mean":
            features = np.mean(features, axis=0)
        return features
    
    else:
        return None


def pad_features(features, max_len=400, pad_value=0.):
    h, w = features.shape
    if h >= max_len:
        return features[:max_len, :], max_len
    else:
        pad_array = np.full((max_len - h, w), pad_value)
        features = np.concatenate((features, pad_array), axis=0)
        return features, h


class EmotionDataset(Dataset):
    def __init__(self, metadata_path, max_len=400, config=None, library="librosa"):
        super().__init__()
        self.config = config
        self.max_len = max_len
        self.data = pd.read_csv(metadata_path)
        self.y = np.array(self.data["label"])
        wav_paths = list(self.data["path"])
        self.X, self.X_lengths = self.load_features(wav_paths)
        
    def load_features(self, wav_paths):
        waveforms = []
        waveform_lengths = []
        for wav_path in wav_paths:
            wav = load_waveform(wav_path, sr=self.config["sr"])
            features = extract_features(wav, feature_type=self.config["features"], config=self.config)
            features, feature_length = pad_features(features, max_len=self.max_len)
#             print(features.shape)
            waveforms.append(features)
            waveform_lengths.append(feature_length)
        waveforms = np.stack(waveforms)
        waveform_lengths = np.array(waveform_lengths)
        return waveforms, waveform_lengths

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        input = self.X[idx]
        input_length = int(self.X_lengths[idx])
        label = int(self.y[idx])
        input = torch.tensor(input, dtype=torch.float)
        return input, input_length, label


class LSTMModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers=2, 
                 hidden_dim=128, bidirectional=True, dropout_rate=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                            bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, inputs, input_lengths):
        batch_size, max_len, _ = inputs.size()
#         inputs = torch.nn.utils.rnn.pack_padded_sequence(
#         inputs, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        embeddings, (h, c) = self.lstm(inputs)
# #         embeddings, _ = torch.nn.utils.rnn.pad_packed_sequence(embeddings, batch_first=True, 
# #                                                                padding_value=0.0, total_length=max_len)
#         embeddings = self.dropout(embeddings)
        # outputs = self.fc(embeddings)
        h = torch.cat((h[-2,:,:], h[-1,:,:]), dim = 1)
        h = self.dropout(h)
        outputs = self.fc(h)
        return outputs


train_data = EmotionDataset(METADATA_TRAIN_PATH, config=config)
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=64,
                                           shuffle=True, num_workers=8)

valid_data = EmotionDataset('./datasets/data_ravdess_savee_valid_test.csv', config=config)
valid_loader = torch.utils.data.DataLoader(valid_data,
                                           batch_size=64,
                                           shuffle=False, num_workers=8)



# Reference: https://github.com/kuangliu/pytorch-cifar
term_width = 25
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f



device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LSTMModel(input_dim=config["n_mels"], num_classes=len(INDICES_TO_EMOTIONS), 
                  num_layers=2, hidden_dim=128, bidirectional=True, dropout_rate=0.5)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

best_acc = 0
start_epoch = 0 
epochs = 200


def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, input_lengths, labels) in enumerate(train_loader):
        inputs, input_lengths, labels = inputs.to(device), input_lengths.to(device), labels.to(device)
        outputs = model(inputs, input_lengths)
        loss = criterion(outputs, labels)
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    return train_loss/(batch_idx+1), 100.*correct/total


def valid(epoch):
    global best_acc
    model.eval()
    valid_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, input_lengths, labels) in enumerate(valid_loader):
            inputs, input_lengths, labels = inputs.to(device), input_lengths.to(device), labels.to(device)
            outputs = model(inputs, input_lengths)
            loss = criterion(outputs, labels)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            progress_bar(batch_idx, len(valid_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (valid_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
    return valid_loss/(batch_idx+1), 100.*correct/total


train_losses = []
valid_losses = []
for epoch in range(start_epoch, epochs):
    train_loss, train_acc = train(epoch)
    valid_loss, test_acc = valid(epoch)
    best_acc = max(test_acc, best_acc)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.9 * param_group['lr']