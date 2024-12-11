# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:58:11 2024

@author: jarom
"""
import os
import numpy as np
import pandas as pd
import scipy.io
from scipy.io import wavfile
from scipy.signal import decimate

# paths
DATA_DIR = "D:/orthop/insect_sounds"
OUT_DATA_DIR = "D:/orthop/insect_sounds/data/InsectSetDownsampledSortedCut"

# params
DOWNSAMPLE_FACTOR = 2
OUT_FILE_LENGTH_S = 5

base_path = "D:/orthop/insect_sounds"
insect_set66_file = os.path.join(base_path, "data/InsectSet/audio_files_InsectSet66.csv")

df_insect66 = pd.read_csv(insect_set66_file)

"""
data preparation for "vanilla" ECOGEN training

from the ECOGEN github repo:

1. Convert the audio files to mono channel
    they already are
    
2. Resample the audio files to 22050 Hz
    downsample by factor 2 using scipy.signal.decimate
    this is not ideal. nyquist frequency is 11025 which is too low for many orthoptera species
    TODO (many species are active in higher frequencies. this would have to be adapted...)

3. Trim the audio files to 5 seconds
    files > 5s: discard
    TODO could pad with noise later
    files 10 > 5s: take middle 5s
    files longer than 10s: cut into 5s
    TODO could add overlap later

training code expects files:
    ./birds-songs/dataset/train.txt|test.txt
with the file paths (the labels are expected to be part of the path)
    birds-song/1.wav

"""

def process_audio_file(row):
    directory = row["directory"]
    fname_orig = row["file_name"]
    species = row["species"]
    subset = row["subset"]
    
    audio_fpath = os.path.join(DATA_DIR, directory, fname_orig)
    
    """
    store in this folder structure:
    train
    test
    validate
    - Roeselianaroeselii
    - ...
    """
    out_dir = os.path.join(OUT_DATA_DIR, subset, species)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # load audio file and downsample
    # InsectSet66 files are in 16-bit integer PCM, we will get dtype int16
    # for orig_wave
    orig_sample_rate, orig_wave = wavfile.read(audio_fpath)
    
    assert orig_sample_rate == 44100, f"Sample rate not 44100, but {orig_sample_rate}"
    
    
    new_sample_rate = orig_sample_rate // DOWNSAMPLE_FACTOR    
    downsampled_wave = decimate(orig_wave, DOWNSAMPLE_FACTOR)
    
    # convert back to the same data type
    downsampled_wave = downsampled_wave.astype("int16")
    
    # handle audio length
    wave_length = len(downsampled_wave) / new_sample_rate
    
    if wave_length < OUT_FILE_LENGTH_S:
        print(f"Too short, discarded: {fname_orig}")
        return None
    
    # file is 5 to 10 seconds long, trim to 5s
    elif OUT_FILE_LENGTH_S <= wave_length < 2*OUT_FILE_LENGTH_S:
        downsampled_wave = downsampled_wave[:new_sample_rate*OUT_FILE_LENGTH_S]
        out_fpath = os.path.join(OUT_DATA_DIR, subset, species, fname_orig)
        wavfile.write(out_fpath, new_sample_rate, downsampled_wave)
        assert len(downsampled_wave) == new_sample_rate*OUT_FILE_LENGTH_S, "illegal length"
        assert os.path.isfile(out_fpath), f"Failed writing {out_fpath}"
        
    # file is equal or longer than 10s, we can make num_files from this
    # TODO could add overlap later, to generate more samples
    elif wave_length >= 2*OUT_FILE_LENGTH_S:
        num_files = int(wave_length // OUT_FILE_LENGTH_S)
        print(f"File {fname_orig} is longer than 10s, cutting into {num_files} new files.")
        for i in range(num_files):
            start = i*new_sample_rate*OUT_FILE_LENGTH_S
            end = (i+1)*new_sample_rate*OUT_FILE_LENGTH_S
            out_file_cut = downsampled_wave[start:end]
            assert len(out_file_cut) == new_sample_rate*OUT_FILE_LENGTH_S, "illegal length"
            
            fname = fname_orig.replace(".wav", f"_cut_{i}.wav")
            out_fpath = os.path.join(OUT_DATA_DIR, subset, species, fname)
            wavfile.write(out_fpath, new_sample_rate, out_file_cut)
            assert os.path.isfile(out_fpath), f"Failed writing {out_fpath}"
            
        # this is whacky i'm sorry
        out_fpath = out_fpath.replace(".wav", f"cut_into_{num_files}.wav")
    else:
        # just to check if there are mistakes in the if-elif-else logic
        print("this should never happen.")
        print(f"{wave_length=}")
        out_fpath = None
       
    return out_fpath
    

# this will generate the downsampled and cut sound files, and sort them according
# to species and subset (train test validate)
df_insect66["downsampled_fpath"] = df_insect66.apply(process_audio_file, axis=1)


# df_insect66.to_excel("InsectSetDownsampledSorted.xlsx")

# have to put the cut files back into the df
# add rows for files we cut up
# remove files we discarded because they were too short
df_insect66 = df_insect66[df_insect66["downsampled_fpath"].notna()]


# the filename references for the cut files are now unfortunately messed up
# next time think ahead before writing a bunch of code
# to avoid running the the entire cutting process again we recover them as 
# follows:

import re
# example:
# messed up file path string:
# Chorthippusalbomarginatus_XC751395-dat022-012_edit2_cut_1cut_into_2.wav

# original file name
# Chorthippusalbomarginatus_XC751395-dat022-012_edit2.wav

# cleaned up:
# Chorthippusalbomarginatus_XC751395-dat022-012_edit2_cut_1.wav
# Chorthippusalbomarginatus_XC751395-dat022-012_edit2_cut_2.wav

def recover_fnames(row):
    pattern = r"_cut_\d+cut_into_(\d+)"
    matches = re.findall(pattern, row["downsampled_fpath"])
    num_cuts = int(matches[0])
    fname_orig = row["file_name"]
    subset = row["subset"]
    species = row["species"]
    
    new_fnames = [fname_orig.replace(".wav", f"_cut_{i}.wav") for i in range(num_cuts)]
    new_outfpaths = [os.path.join(OUT_DATA_DIR, subset, species, fname) for fname in 
                     new_fnames]
    
    assert all([os.path.isfile(file) for file in new_outfpaths]), f"File renamed wrongly {fname_orig}"
    
    new_rows = [{'insect_set': row['insect_set'], 
                 'directory': row['directory'], 
                 'file_name': row["file_name"],
                 'species': row["species"],
                 'subset': row["subset"],
                 'unique_file': row["unique_file"],
                 'link': row["link"],
                 'contributor': row["contributor"],
                 'bits_per_sample': row["bits_per_sample"],
                 'encoding': row["encoding"],
                 'num_channels': row["num_channels"],
                 'num_frames': row["num_frames"],
                 'sample_rate': row["sample_rate"],
                 'audio_length': row["audio_length"],
                 'class': row["class"],
                 'downsampled_fpath': new_fpath} for new_fpath in new_outfpaths]
    
    return new_rows

# get the row with files that were cut into multiple files
condition = df_insect66["downsampled_fpath"].str.contains("cut_into_")
rows_to_duplicate = df_insect66[condition].copy()

# drop the rows we want to duplicate from the original df
df_insect66 = df_insect66[~condition]

# recover the file names using the function
new_rows = rows_to_duplicate.apply(recover_fnames, axis=1).sum()
new_df = pd.DataFrame(new_rows)

# put the recovered rows back into the original df
df_insect66 = pd.concat([df_insect66, new_df])

# the durations in the df are not right. "quick" sanity checks:
import wave
def check_durations_cut_files(row):
    with wave.open(row["downsampled_fpath"]) as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
    
    return pd.Series([frames, duration, rate], index=["num_frames", "audio_length", "sample_rate"])

df_insect66_bak = df_insect66.copy()
df_insect66[["num_frames", "audio_length", "sample_rate"]] = df_insect66.apply(check_durations_cut_files, axis=1)

assert all(df_insect66["audio_length"]==5.0), "Not all audios are 5s"
assert all(df_insect66["sample_rate"]==22050), "Not all audios are 22.05kHz sample rate"

# replace "D:/orthop/insect_sounds" with "."
df_insect66["downsampled_fpath"] = df_insect66["downsampled_fpath"].str.replace("D:/orthop/insect_sounds", ".")
df_insect66["downsampled_fpath"] = df_insect66["downsampled_fpath"].str.replace("\\", "/")

# split up for txt files
df_test = df_insect66[df_insect66["subset"]=="test"]
df_val = df_insect66[df_insect66["subset"]=="validation"]

# to txt files
"""
training code expects files:
    ./birds-songs/dataset/train.txt|test.txt
with the file paths:
    birds-song/1.wav
"""
subsets = ["train", "test", "validation"]

for subset in subsets:
    df_out = df_insect66[df_insect66["subset"]==subset]
    
    fpaths = df_out["downsampled_fpath"]
    
    # Write the column to a text file
    with open(f'{subset}.txt', 'w') as file:
        file.write('\n'.join(fpaths.astype(str)))

df_insect66.to_excel("insect66cut.xlsx")


























# 
