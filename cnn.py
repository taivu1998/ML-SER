import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft 
from scipy import signal
from scipy.io import wavfile

import sklearn
from sklearn.model_selection import train_test_split
from pathlib import Path
from IPython.display import Audio
import librosa
import librosa.display

import fastai
from fastai.vision import *
from fastai.vision.all import *


METADATA_DIR = './datasets/data_ravdess_savee.csv'
METADATA_TRAIN_PATH = './datasets/data_ravdess_savee_train.csv'
METADATA_VALID_PATH = './datasets/data_ravdess_savee_valid.csv'
METADATA_TEST_PATH = './datasets/data_ravdess_savee_test.csv'

METADATA_DIR = './datasets/metadatas'
IMAGE_DIR = './datasets/images'


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
        # eps = 1e-10
        # features = np.log(features.T + eps)
        features = librosa.power_to_db(features, ref=np.max)
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


def load_waveform(wav_path, sr=22050, library="librosa"):
    wav, sr = librosa.load(wav_path, sr)
    wav, _ = librosa.effects.trim(wav)
    return wav


def pad_features(features, max_len=400, pad_value=0.):
    h, w = features.shape
    if h >= max_len:
        return features[:max_len, :]
    else:
        pad_array = np.full((max_len - h, w), pad_value)
        features = np.concatenate((features, pad_array), axis=0)
        return features


def create_metadata_and_images(config, metadata_path, image_dir):
    df_train = pd.read_csv(METADATA_TRAIN_PATH)
    df_valid = pd.read_csv(METADATA_VALID_PATH)
    df_train['is_valid'] = False
    df_valid['is_valid'] = True
    df = pd.concat([df_train, df_valid])
    df = df.reset_index()

    img_paths = []
    for i in range(len(df)):
        path = df.iloc[i]["path"]
        image_path = path[path.rfind("/") + 1:path.rfind(".")] + ".png"
        image_path = os.path.join(image_dir, image_path)
        img_paths.append(image_path)
        wav = load_waveform(path, sr=config["sr"])
        features = extract_features(wav, feature_type=config["features"], config=config)
        # features = pad_features(features, max_len=400)
        # features = features.T
        # plt.imsave(image_path, features)
        librosa.display.specshow(features, fmax=20000, x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.savefig(image_path)
    df["img_path"] = img_paths
    df.to_csv(metadata_path, index=False)
    return df


def read_metadata(config):
    metadata_file = "features={}_normalize={}_sr={}_nmels={}_nfft={}_hoplength={}_winlength={}_wavmaxlength={}_nmfcc={}_augment={}.csv".format(
        config["features"], config["normalize"], config["sr"], config["n_mels"], config["n_fft"], 
        config["hop_length"], config["win_length"], config["wav_max_length"], config["n_mfcc"], config["augment"]
    )
    metadata_path = os.path.join(METADATA_DIR, metadata_file)

    if os.path.isfile(metadata_path):
        df = pd.read_csv(metadata_path)
        return df
    else:
        image_dir_name = metadata_file[:-4]
        image_dir = os.path.join(IMAGE_DIR, image_dir_name)
        if not os.path.isdir(image_dir):
            os.makedirs(image_dir)
        df = create_metadata_and_images(config, metadata_path, image_dir)
        return df


df = read_metadata(config)
tfms = aug_transforms(do_flip=False, max_rotate=5, max_zoom=1.05, max_lighting=0.1, max_warp=0)
nrm = Normalize.from_stats(*imagenet_stats, cuda=True)
dls = ImageDataLoaders.from_df(df, fn_col='img_path', label_col='label', valid_col='is_valid',
                               item_tfms=Resize(128), batch_tfms=[*tfms, nrm], num_workers=4)
dls.show_batch(max_n=9)


learn = cnn_learner(dls, resnet34, metrics=accuracy, cbs=MixUp())


for i in range(6):
    lr = learn.lr_find().lr_min
    learn.fit_one_cycle(5, lr)


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(10, 10), dpi=60)


tfms = aug_transforms(do_flip=False, max_rotate=5, max_zoom=1.05, max_lighting=0.1, max_warp=0)
nrm = Normalize.from_stats(*imagenet_stats, cuda=True)
dls = ImageDataLoaders.from_df(df, fn_col='img_path', label_col='label', valid_col='is_valid',
                               item_tfms=Resize(256), batch_tfms=[*tfms, nrm], num_workers=4)
learn.dls = dls


learn.freeze()
for i in range(6):
    lr = learn.lr_find().lr_min
    learn.fine_tune(5, lr)


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(10, 10), dpi=60)