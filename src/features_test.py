import librosa
import cv2
import numpy as np
import torch

def load_audio(file_path, sr, window_length=0):
    audio, _sr = librosa.load(file_path, sr=sr)
    print(sr)
    """
    if _sr != self.sr:
        audio = librosa.resample(y=audio, orig_sr=_sr, target_sr=sr)

    if self.window_length and len(audio) >= self.window_length:
        audio = audio[0:self.window_length]
    else:
        audio = librosa.util.fix_length(data=audio, size=self.window_length)
    """
    return audio

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


audio = load_audio(file_path=r"D:\orthop\insect_sounds\data\InsectSetDownsampledSortedCut\train\Achetadomesticus\Achetadomesticus_XC489192-Achetadomesticus_poland_psz_20140510_22.00h_3498_edit1.wav", sr=22050)

features = audio_to_melspectrogram(audio)

features = np.expand_dims(features,0)

features = torch.tensor(features)


print(features.shape)