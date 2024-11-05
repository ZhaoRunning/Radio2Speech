from torch.utils.data import Dataset
import torch

import matplotlib.pyplot as plt
from mel_Arguments import ArgParser
import librosa.display
import librosa
import numpy as np
import h5py
import csv
import math

def trans_list(input_csv):
    list_sample = []
    for row in csv.reader(open(input_csv, 'r'), delimiter=','):
        if len(row) < 2:
            continue
        list_sample.append(row)
    print(len(list_sample))
    return list_sample

class Meldataset(Dataset):
    def __init__(self, input_path=None):
        """
        files should be a list [(radio file, audio file)]
        """

        self.files = trans_list(input_path)
        self.expmel_len = 80 

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        info = self.files[index]
        radio_file = info[0]
        audio_file = info[1]
        mel_len = int(info[2])

        # audio load
        audio_h5f = h5py.File(audio_file, 'r')
        audio_melamp = audio_h5f['mel'][:]  # mel / feats
        # radio load
        raido_h5f = h5py.File(radio_file, 'r')
        radio_melamp = raido_h5f['mel'][:]

        assert audio_melamp.shape == radio_melamp.shape
        assert mel_len == audio_melamp.shape[0]

        # random cut
        if mel_len > self.expmel_len:
            center = np.random.randint(self.expmel_len // 2, mel_len - self.expmel_len // 2)
            start = max(0, center - self.expmel_len // 2)
            end = min(mel_len, center + self.expmel_len // 2)
            audio_melamp = audio_melamp[start:end]
            radio_melamp = radio_melamp[start:end]
        elif mel_len < self.expmel_len:
            # tile the audio and radio
            n = int(self.expmel_len / mel_len) + 1
            audio_melamp = np.tile(audio_melamp, (n, 1))
            audio_melamp = audio_melamp[:self.expmel_len, :]
            radio_melamp = np.tile(radio_melamp, (n, 1))
            radio_melamp = radio_melamp[:self.expmel_len, :]
            # pad the audio and radio
            # pad_len = self.expmel_len - mel_len
            # audio_melamp = np.pad(audio_melamp, ((0, pad_len), (0, 0)))
            # radio_melamp = np.pad(radio_melamp, ((0, pad_len), (0, 0)))

        return torch.FloatTensor(audio_melamp).unsqueeze(0), torch.FloatTensor(radio_melamp).unsqueeze(0)

class Meldataset_sliding:
    def __init__(self, input_list=None, tile=True, expmel_len=128, stride=64):
        """
        files should be a list [(radio file, audio file)]
        """

        self.files = input_list
        self.tile = tile

        self.audio_length = expmel_len
        self.audio_stride = stride
        self.num_examples = []

        for _, _, file_length in self.files:
            file_length = int(file_length)
            if self.audio_length is None:
                examples = 1
            elif file_length < self.audio_length:
                examples = 1 if self.tile else 0
            elif self.tile:
                examples = int(math.ceil((file_length - self.audio_length) / self.audio_stride) + 1)
            else:
                examples = (file_length - self.audio_length) // self.audio_stride + 1
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        for (radio_file, audio_file, _), examples in zip(self.files, self.num_examples):
            if index >= examples:
                index -= examples
                continue

            start = self.audio_stride * index
            num_frames = self.audio_length

            # audio load
            audio_h5f = h5py.File(audio_file, 'r')
            audio_melamp = audio_h5f['feats'][:]
            # radio load
            raido_h5f = h5py.File(radio_file, 'r')
            radio_melamp = raido_h5f['mel'][:]

            assert audio_melamp.shape == radio_melamp.shape

            # cut in fixed length
            audio_melamp = audio_melamp[start : start + num_frames]
            radio_melamp = radio_melamp[start : start + num_frames]

            if audio_melamp.shape[0] < self.audio_length:
                # tile the audio and radio
                n = int(self.audio_length / audio_melamp.shape[0]) + 1
                audio_melamp = np.tile(audio_melamp, (n, 1))
                audio_melamp = audio_melamp[:self.audio_length, :]
                radio_melamp = np.tile(radio_melamp, (n, 1))
                radio_melamp = radio_melamp[:self.audio_length, :]

            return torch.FloatTensor(audio_melamp).unsqueeze(0), torch.FloatTensor(radio_melamp).unsqueeze(0)


class Meldataset2(Dataset):
    def __init__(self, input_path, tile=True, expmel_len=128, stride=64):
        """__init__.
        :param input_path: directory containing both clean.json and noisy.json
        """
        self.list_input = trans_list(input_path)
        self.radioaudio_set = Meldataset_sliding(self.list_input,
                                                 tile=tile, expmel_len=expmel_len, stride=stride)

    def __getitem__(self, index):
        audio_amp, radio_amp = self.radioaudio_set[index]
        return audio_amp, radio_amp

    def __len__(self):
        return len(self.radioaudio_set)
