import os
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


class SpeechDataset():
    def __init__(self, metadata_train_path, metadata_valid_path, metadata_test_path,
                 sr=22050, library="librosa"):
        self.data_train = pd.read_csv(metadata_train_path)
        self.data_valid = pd.read_csv(metadata_valid_path)
        self.data_test = pd.read_csv(metadata_test_path)
        
        self.y_train = self.data_train["label"]
        self.X_train = self.data_train.drop("label", axis=1)
        self.X_train_paths = list(self.data_train["path"])
        self.y_valid = list(self.data_valid["label"])
        self.X_valid = self.data_valid.drop("label", axis=1)
        self.X_valid_paths = list(self.data_valid["path"])
        self.y_test = self.data_test["label"]
        self.X_test = self.data_test.drop("label", axis=1)
        self.X_test_paths = list(self.data_test["path"])
        
        self.X_train_waveforms = self.load_waveforms(self.X_train_paths)
        self.X_valid_waveforms = self.load_waveforms(self.X_valid_paths)
        self.X_test_waveforms = self.load_waveforms(self.X_test_paths)
    
    def load_waveforms(self, wav_paths, sr=22050, library="librosa"):
        waveforms = [load_waveform(wav_path, sr=sr, library=library) for wav_path in wav_paths]
        return waveforms
        
    def get_train_data(self):
        return self.X_train_waveforms, self.y_train
    
    def get_valid_data(self):
        return self.X_valid_waveforms, self.y_valid
    
    def get_test_data(self):
        return self.X_test_waveforms, self.y_test


dataset = SpeechDataset(METADATA_TRAIN_DIR, METADATA_VALID_DIR, METADATA_TEST_DIR)
X_train, y_train = dataset.get_train_data()
X_valid, y_valid = dataset.get_valid_data()
X_test, y_test = dataset.get_test_data()


class FeatureExtractor():
    def __init__(self, X_train, X_valid, X_test, 
                 feature_type="mfcc", config=None, 
                 reduction=None, library="librosa"):
        self.X_train_features = self.extract_input_features(X_train, feature_type=feature_type, 
                                                            config=config, reduction=reduction, 
                                                            library=library)
        self.X_valid_features = self.extract_input_features(X_valid, feature_type=feature_type, 
                                                            config=config, reduction=reduction, 
                                                            library=library)
        self.X_test_features = self.extract_input_features(X_test, feature_type=feature_type, 
                                                           config=config, reduction=reduction, 
                                                           library=library)
    
    def extract_input_features(self, waveforms, feature_type="mfcc", config=None, reduction=None, library="librosa"):
        input_features = [extract_features(wav, feature_type=feature_type, 
                                           config=config, reduction=reduction, 
                                           library=library) for wav in waveforms]
        return input_features 
    
    def get_train_features(self):
        return self.X_train_features
    
    def get_valid_features(self):
        return self.X_valid_features
        
    def get_test_features(self):
        return self.X_test_features


feature_extractor = FeatureExtractor(X_train, X_valid, X_test, 
                                     feature_type="mfcc", config=config, 
                                     reduction="mean", library="librosa")


X_train = feature_extractor.get_train_features()
X_valid = feature_extractor.get_valid_features()
X_test = feature_extractor.get_test_features()


X_train = np.asarray(X_train)
X_valid = np.asarray(X_valid)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_valid = np.asarray(y_valid)
y_test = np.asarray(y_test)


clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, y_train)
clf.score(X_train, y_train)
clf.score(X_valid, y_valid)