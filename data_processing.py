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


RAVDESS_DIR = './datasets/RAVDESS/Audio_Speech_Actors_01-24/'
RAVDESS_METADATA_DIR = './datasets/ravdess.csv'
RAVDESS_EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust', 
    '08': 'surprised',
}


def read_data(data_dir):
    data_dict = {
        "label": [],
        "gender": [],
        "path": [],
    }

    for actor_idx in range(1, 25):
        actor_dir = os.path.join(data_dir, f'Actor_{str(actor_idx).zfill(2)}')
        for file_name in os.listdir(actor_dir):
            file_path = os.path.join(actor_dir, file_name)
            file_name_parts = file_name.split('.')[0]
            file_name_parts = file_name_parts.split("-")
            emotion = RAVDESS_EMOTION_MAP[file_name_parts[2]]
            gender = int(file_name_parts[6])
            emotion_idx = EMOTIONS_TO_INDICES[emotion]
            gender_idx = gender % 2
            data_dict["label"].append(emotion_idx)
            data_dict["gender"].append(gender_idx)
            data_dict["path"].append(file_path)
    df = pd.DataFrame.from_dict(data_dict)
    return df


def create_metadata(data_dir, metadata_path):
    df = read_data(data_dir)
    df.to_csv(metadata_path, index=False)


create_metadata(RAVDESS_DIR, RAVDESS_METADATA_DIR)


SAVEE_DIR = './datasets/SAVEE/AudioData/'
SAVEE_METADATA_PATH = './datasets/savee.csv'
SAVEE_EMOTION_MAP = {
    'n': 'neutral',
    'h': 'happy',
    'sa': 'sad',
    'a': 'angry',
    'f': 'fearful',
    'd': 'disgust', 
    'su': 'surprised',
}


def read_data(data_dir):
    data_dict = {
        "label": [],
        "gender": [],
        "path": [],
    }
    for actor_name in ['DC', 'JE', 'JK', 'KL']:
        actor_dir = os.path.join(data_dir, actor_name)
        for file_name in os.listdir(actor_dir):
            file_path = os.path.join(actor_dir, file_name)
            emotion = SAVEE_EMOTION_MAP[file_name[:-6]]
            emotion_idx = EMOTIONS_TO_INDICES[emotion]
            gender_idx = 1
            data_dict["label"].append(emotion_idx)
            data_dict["gender"].append(gender_idx)
            data_dict["path"].append(file_path)
    df = pd.DataFrame.from_dict(data_dict)
    return df


def create_metadata(data_dir, metadata_path):
    df = read_data(data_dir)
    df.to_csv(metadata_path, index=False)


create_metadata(SAVEE_DIR, SAVEE_METADATA_PATH)


RAVDESS_METADATA_PATH = './datasets/ravdess.csv'
SAVEE_METADATA_PATH = './datasets/savee.csv'
METADATA_PATH = './datasets/data_ravdess_savee.csv'


def combine_datasets(dataset_metadata_paths):
    metadata = [pd.read_csv(dataset_metadata_path) for dataset_metadata_path in dataset_metadata_paths]
    df = pd.concat(metadata)
    df = df.reset_index()
    return df


df = combine_datasets([RAVDESS_METADATA_PATH, SAVEE_METADATA_PATH])
df.to_csv(METADATA_PATH, index=False)


def split_data(metadata_path, metadata_train_path, metadata_valid_path, metadata_test_path,
               valid_size=0.05, test_size=0.05, keep_class_ratios=False):
    metadata = pd.read_csv(metadata_path)
    y = metadata["label"]
    X = metadata.drop("label", axis=1)
    
    if keep_class_ratios: 
        X_train_valid, X_test, y_train_valid, y_test = \
            train_test_split(X, y, test_size=test_size, random_state=0, stratify=y)
        X_train, X_valid, y_train, y_valid = \
            train_test_split(X_train_valid, y_train_valid, test_size=valid_size/(1-test_size), random_state=0, stratify=y_train_valid)
    else:
        X_train_valid, X_test, y_train_valid, y_test = \
            train_test_split(X, y, test_size=test_size, random_state=0)
        X_train, X_valid, y_train, y_valid = \
            train_test_split(X_train_valid, y_train_valid, test_size=valid_size/(1-test_size), random_state=0)
    
    data_train = pd.concat([y_train, X_train], axis=1)
    data_valid = pd.concat([y_valid, X_valid], axis=1)
    data_test = pd.concat([y_test, X_test], axis=1)
    
    data_train.to_csv(metadata_train_path, index=False)
    data_valid.to_csv(metadata_valid_path, index=False)
    data_test.to_csv(metadata_test_path, index=False)


METADATA_PATH = './datasets/data_ravdess_savee.csv'
METADATA_TRAIN_PATH = './datasets/data_ravdess_savee_train.csv'
METADATA_VALID_PATH = './datasets/data_ravdess_savee_valid.csv'
METADATA_TEST_PATH = './datasets/data_ravdess_savee_test.csv'


split_data(METADATA_PATH, METADATA_TRAIN_PATH, 
           METADATA_VALID_PATH, METADATA_TEST_PATH,
           valid_size=0.05, test_size=0.05)