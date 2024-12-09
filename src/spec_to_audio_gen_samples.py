#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 09:28:46 2024

@author: sebastien
"""
import librosa
import matplotlib.pyplot as plt
import numpy as np
import cv2
import soundfile as sf

def mel_spectrogram_to_audio(spectrogram, resize=True):
    if resize:
        if isinstance(resize, set):
            spectrogram = cv2.resize(spectrogram, resize)
        else:
            spectrogram = cv2.resize(spectrogram,(64,128))
    print("Mel shape", spectrogram.shape)
    spectrogram = librosa.db_to_power(spectrogram)
    s = librosa.feature.inverse.mel_to_stft(spectrogram)
    y = librosa.griffinlim(s)
    return y

def audio_to_melspectrogram(audio, resize=True):
    "This function only works for image with one channel"
    features = librosa.feature.melspectrogram(y=audio, n_fft=1024)
    features = librosa.power_to_db(features)
    if resize:
        features = cv2.resize(features,(32,256))
    #If height is odd, skip the first column
    if features.shape[0] % 2 != 0: 
        features = features[1:, :]
    #If row is odd, skip the first row
    if features.shape[1] % 2 != 0:
        features = features[:, 1:]
    return features






working_dir = '/home/sebastien/Downloads/InsectSetDownsampledSortedCut-001/test_interpolation/Achetadomesticus/'
original_wave_path = '/home/sebastien/Downloads/InsectSetDownsampledSortedCut-001/test_interpolation/x/Grylluscampestris_GBIF2626668332_IN45297253_85264_cut_1.wav'
spec_gen_path = working_dir + 'Achetadomesticus_XC751737-dat001-058_cut_0-1_interpolation0.42-Achetadomesticus_XC751737-dat001-058_cut_1.wav.wav.npy'
y, sr = librosa.load(original_wave_path)
spec_gen = np.load(spec_gen_path)

plt.imshow(spec_gen)

# original_spec = np.abs(librosa.stft(y))
original_mel_spec = audio_to_melspectrogram(y, resize=False)
plt.imshow(original_mel_spec)

gen_wave = mel_spectrogram_to_audio(spec_gen, resize=False)

plt.plot(gen_wave)
plt.plot(y)


sf.write(working_dir + 'generation_intepol.wav', gen_wave, 22050)



