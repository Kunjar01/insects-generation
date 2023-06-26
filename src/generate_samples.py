from copy import deepcopy
import os
import random
import pytorch_lightning as pl
import logging
import os
from scipy import rand

from torch._C import device
os.environ['HYDRA_FULL_ERROR'] = '1'
from argparse import Namespace
import torch
import hydra
from omegaconf import DictConfig
import argparse
import glob
try:
    from train_vqvae import VQEngine
except ImportError:
    from src.train_classifier import VQEngine
import tqdm
import numpy as np

parser = argparse.ArgumentParser(description="Data Augmentator")
parser.add_argument('--model_path')
# parser.add_argument('--data_paths', type=str, default="/Users/test/Documents/Projects/Master/nips4bplus/train/Regign_song/*.npy", help="Spectrogram(2d numpy) path list.")
parser.add_argument('--data_paths', type=str, default="/Users/test/Documents/Projects/Master/nips4bplus/splits_new/cleaned_train/*/*.npy", help="Spectrogram(2d numpy) path list.")
parser.add_argument('--augmentations', default="noise")
parser.add_argument('--num_samples', type=int, default= 10)
parser.add_argument('--device', default='cpu')

class Augmentations():
    all_methods = [
        'noise',
        'interpolation',
        # 'extrapolation',
        ]

    #Fix: Similar Generate samples are overwritten

    @classmethod
    def encode(self, model, img, device='cuda:0'):
        img = img.to(device)
        quant_top, quant_bottom, diff, id_top, id_bottom = model.encode(img)
        return (quant_top,
            quant_bottom,
            id_top,
            id_bottom)

    def load_sample(filepath):
        spectrogram = np.load(filepath)
        spectrogram = np.pad(spectrogram,[(0,0), (0,4)], mode='edge')
        spectrogram = np.expand_dims(spectrogram, 0)
        spectrogram = np.expand_dims(spectrogram, 0)
        return spectrogram


    @classmethod
    def decode(self, model, quant_top, quant_bottom):
        return model.decode(quant_top, quant_bottom).detach()

    @classmethod
    def noise(self, model, all_samples_paths, ratio=0.5, generation_count=10, device='cpu'):
        all_samples_paths = [x for x in all_samples_paths if 'noise' not in os.path.basename(x)]

        for i, sample_path in tqdm.tqdm(enumerate(all_samples_paths)):
            for j in range(generation_count):

                ratio = random.random() #Generate number in [0,1)
                spectrogram = torch.tensor(self.load_sample(sample_path))
                q_t, q_b, i_t, i_b = self.encode(model.net, spectrogram, device=device)
                new_q_t = ratio*torch.randn_like(q_t) + q_t
                new_q_b = ratio*torch.randn_like(q_b) + q_b
                reconstructed = self.decode(model.net, new_q_t, new_q_b).cpu().numpy()[0][0]
                reconstructed = reconstructed[:,:-4]

                filename, ext = os.path.splitext(sample_path)
                outfile = f"{filename}-{j}_noise{ratio:.2f}{ext}"
                np.save(outfile, reconstructed)


    @classmethod
    def interpolation(self, model, all_samples_paths, ratio=0.5, generation_count=10, device='cpu'):
        all_samples_paths = [x for x in all_samples_paths if 'interpolation' not in os.path.basename(x)]

        for i, sample_path in tqdm.tqdm(enumerate(all_samples_paths)):
            count = 0
            tmp_all_samples_paths = deepcopy(all_samples_paths)
            random.shuffle(tmp_all_samples_paths)

            for k, sample_path1 in enumerate(tmp_all_samples_paths):
                if count >= generation_count:
                    break
                if sample_path == sample_path1:
                        continue
                sample_class = sample_path.split(os.sep)[-2]
                sample_class1 = sample_path1.split(os.sep)[-2]
                if sample_class == sample_class1:
                    ratio = random.random() #Generate number in [0,1)
                    spectrogram = torch.tensor(self.load_sample(sample_path))
                    spectrogram1 = torch.tensor(self.load_sample(sample_path1))

                    q_t, q_b, i_t, i_b = self.encode(model.net, spectrogram, device=device)
                    q_t1, q_b1, i_t1, i_b1 = self.encode(model.net, spectrogram1, device=device)
                    new_q_t = (q_t1 - q_t)*ratio + q_t
                    new_q_b = (q_b1 - q_b)*ratio + q_b

                    reconstructed = self.decode(model.net, new_q_t, new_q_b).cpu().numpy()[0][0]
                    reconstructed = reconstructed[:,:-4]

                    filename, ext = os.path.splitext(sample_path)
                    outfile = f"{filename}-{k}_interpolation{ratio:.2f}-{os.path.basename(sample_path1)}{ext}"
                    np.save(outfile, reconstructed)
                    count += 1


    # @classmethod
    # def extrapolation(self, model, all_samples_paths, ratio=0.5, generation_count=10, device='cpu'):
    #     all_samples_paths = [x for x in all_samples_paths if 'extrapolation' not in os.path.basename(x)]

    #     count = 0
    #     for i, sample_path in tqdm.tqdm(enumerate(all_samples_paths)):
    #         for j in range(generation_count):
    #             for k, sample_path1 in enumerate(all_samples_paths):
    #                 if sample_path == sample_path1:
    #                     continue

    #                 ratio = random.rand(0,1)
    #                 spectrogram = torch.tensor(self.load_sample(sample_path))
    #                 spectrogram1 = torch.tensor(self.load_sample(sample_path1))

    #                 q_t, q_b, i_t, i_b = self.encode(model.net, spectrogram, device=device)
    #                 q_t1, q_b1, i_t1, i_b1 = self.encode(model.net, spectrogram1, device=device)
    #                 new_q_t = (q_t - q_t1)*ratio + q_t
    #                 new_q_b = (q_b - q_b1)*ratio + q_b

    #                 reconstructed = self.decode(model.net, new_q_t, new_q_b).cpu().numpy()[0][0]
    #                 reconstructed = reconstructed[:,:-4]

    #                 filename, ext = os.path.splitext(sample_path)
    #                 outfile = f"{filename}-{k}_extrapolation{ratio}-{os.path.basename(sample_path1)}{ext}"
    #                 np.save(outfile, reconstructed)


def main() -> None:
    args = parser.parse_args()
    model = VQEngine.load_from_checkpoint(args.model_path).to(args.device)
    all_samples_paths = glob.glob(args.data_paths)
    aug_methods_names = args.augmentations.split(',')
    for aug_method_name in aug_methods_names:
        if aug_method_name not in Augmentations.all_methods:
            raise NotImplementedError
        func = getattr(Augmentations, aug_method_name)
        func(model=model, all_samples_paths=all_samples_paths, ratio=0.5, generation_count=int(args.num_samples), device=args.device)
    


if __name__ == "__main__":
    main()