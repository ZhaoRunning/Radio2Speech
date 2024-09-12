import torch
from torch.utils.data import Dataset

import numpy as np
import h5py
import csv
import logging
import os
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

import librosa
import librosa.display
import soundfile as sf
import pysepm
from pesq import pesq

import configargparse
from utils.mel_utils import AverageMeter
from parallel_wavegan.utils import load_model
from cnn_transformer.transunet import TransUnet as TransUnet

class ArgParser(object):
    def __init__(self):
        parser = configargparse.ArgumentParser(
            description="Evaluate Radio2Speech to recover speech from radio signal",
            config_file_parser_class=configargparse.YAMLConfigFileParser,  # decide the config file syntax
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        )
        # general configuration
        parser.add("--config", is_config_file=True, help="config file path")


        # Transformer Unet setting
        parser.add_argument('--hidden_size', help='hidden size', type=int)
        parser.add_argument('--transformer_num_layers', help='transformer number of layers', type=int)
        parser.add_argument('--mlp_dim', help='transformer mlp dim', type=int)
        parser.add_argument('--num_heads', help='transformer number of heads', type=int)
        parser.add_argument('--transformer_dropout_rate', help='transformer dropout rate', type=float)
        parser.add_argument('--transformer_attention_dropout_rate', help='transformer attention dropout rate', type=float)

        parser.add_argument('--audRate', help='audio sample rate', type=int)

        # vocoder configuration
        parser.add_argument('--vocoder_ckpt', help='checkpoint path to recover vocoder')
        parser.add_argument('--vocoder_config', help='vocoder configuration')

        #  evaluation index file
        parser.add_argument('--dataset_name', help='dataset name')
        parser.add_argument('--list_val', help='list of validation data')
        parser.add_argument('--audio_path', help='path of raw audio files')
        parser.add_argument('--load_best_model', help='load from best model')
        parser.add_argument('--save_wave_path', help='path to save generated wave files', type=str)

        self.parser = parser

    def parse_train_arguments(self):
        args = self.parser.parse_args()
        return args


def get_power(x):
    S = librosa.stft(x, win_length=2048, hop_length=512)
    S = np.log10(np.abs(S)**2 + 1e-8)
    return S

def compute_log_distortion(x_hr, x_pr):
    S1 = get_power(x_hr)
    S2 = get_power(x_pr)
    lsd = np.mean(np.sqrt(np.mean((S1-S2)**2 + 1e-8, axis=0)))
    # return min(lsd, 10.)
    return lsd

def melamp_show(melamp, sr):
    librosa.display.specshow(melamp, sr=sr, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

def trans_list(input_csv):
    list_sample = []
    for row in csv.reader(open(input_csv, 'r'), delimiter=','):
        if len(row) < 2:
            continue
        list_sample.append(row)
    print(len(list_sample))
    return list_sample

class radioaudiomelDataset(Dataset):
    def __init__(self, dataset_name, input_path=None, audio_path=None, sampling_rate=8000):
        """
        files should be a list [(radio file, audio file, length)]
        """
        self.dataset_name = dataset_name
        self.files = trans_list(input_path)
        self.audio_path = audio_path
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        info = self.files[index]
        radio_file = info[0]
        audio_file = info[1]
        mel_len = int(info[2])

        file_folder = None
        # For LJSpeech
        if self.dataset_name == 'LJSpeech':
            filename = os.path.basename(radio_file).split('.')[0]
            audio_raw_path = os.path.join(self.audio_path, f'{filename}.wav')
        else:
            # for TIMIT
            file_split = radio_file.lstrip('/').split('/')
            file_folder = file_split[-2] # MEDR0
            filename = file_split[-1].split('.')[0] # SA1
            audio_raw_path = os.path.join(self.audio_path, file_folder, f'{filename}.wav')

        # audio load
        audio_h5f = h5py.File(audio_file, 'r')
        audio_melamp = audio_h5f['mel'][:]  # feats / mel
        # radio load
        raido_h5f = h5py.File(radio_file, 'r')
        radio_melamp = raido_h5f['mel'][:]
        # raw audio signal load
        audio_raw, sr = librosa.load(audio_raw_path, sr=None, mono=True)

        assert sr == self.sampling_rate
        assert audio_melamp.shape == radio_melamp.shape
        assert mel_len == audio_melamp.shape[0]

        if file_folder is not None:
            file = os.path.join(file_folder, filename)
        else:
            file = filename

        return (file, audio_raw, audio_melamp, \
               torch.FloatTensor(radio_melamp).unsqueeze(0).unsqueeze(0))


def main():
    parser = ArgParser()
    args = parser.parse_train_arguments()

    # 0. dataset
    val_set = radioaudiomelDataset(args.dataset_name, args.list_val, args.audio_path, args.audRate)

    # 1. define radio_audio unet
    mel_generator = TransUnet(args.hidden_size, 
                              args.transformer_num_layers, 
                              args.mlp_dim, 
                              args.num_heads, 
                              args.transformer_dropout_rate, 
                              args.transformer_attention_dropout_rate
                              ).cuda()

    logging.info(f'Loading best model from: {args.load_best_model}')
    pretrained_dict = torch.load(args.load_best_model, map_location='cpu')
    if 'net' in pretrained_dict.keys():
        pretrained_dict = pretrained_dict['net']
    pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
    mel_generator.load_state_dict(pretrained_dict)
    mel_generator.eval().cuda()

    # 2. define neural vocoder
    with open(args.vocoder_config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    vocoder = load_model(args.vocoder_ckpt, config)
    logging.info(f"Loaded model parameters from {args.vocoder_ckpt}.")
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().cuda()

    # 3. start generation and evaluation
    stoi_metric = AverageMeter()
    llr_metric = AverageMeter()
    pesq_metric = AverageMeter()
    lsd_metric = AverageMeter()

    with torch.no_grad(), tqdm(val_set, desc='[decode]') as pbar:
        for idx, (filename, audio_raw, audio_melamp, radio_melamp) in enumerate(pbar):
            # audio_melamp is used for visualization if necessary

            radio_melamp = radio_melamp.cuda()
            pred = mel_generator(radio_melamp)

            #c(Tensor): Local conditioning auxiliary features (T', C).
            pred = pred.squeeze(0).squeeze(0)
            audio_pred = vocoder.inference(pred, normalize_before=False).view(-1)

            audio_pred = audio_pred.cpu().numpy()
            if audio_pred.shape[0] > audio_raw.shape[0]:
                audio_pred = audio_pred[: audio_raw.shape[0]]

            ## save the predicted wave
            # for TIMIT
            # if '/' in filename:
            #     folder = filename.split('/')[0]
            #     file = filename.split('/')[1]
            #     if not os.path.exists(os.path.join(args.save_wave_path, folder)):
            #         os.makedirs(os.path.join(args.save_wave_path, folder))
            #     sf.write(os.path.join(args.save_wave_path, folder, f"{file}.wav"), audio_pred, args.audRate)
            # else:
            #     # for LJSpeech
            #     if not os.path.exists(os.path.join(args.save_wave_path)):
            #         os.makedirs(os.path.join(args.save_wave_path))
            #     sf.write(os.path.join(args.save_wave_path, f"{filename}.wav"), audio_pred, args.audRate)

            # calculate the metrics
            stoi = pysepm.stoi(audio_raw, audio_pred, args.audRate)
            llr = pysepm.llr(audio_raw, audio_pred, args.audRate)
            pesq_mos_lqo = pesq(args.audRate, audio_raw, audio_pred, 'nb')
            lsd = compute_log_distortion(audio_raw, audio_pred)

            stoi_metric.update(stoi)
            llr_metric.update(llr)
            pesq_metric.update(pesq_mos_lqo)
            lsd_metric.update(lsd)

        print('Evaluation Summary: LSD:{:.4f}, stoi:{:.4f}, llr:{:.4f}, pesq:{:.4f}'
              .format(lsd_metric.average(), stoi_metric.average(),
                      llr_metric.average(), pesq_metric.average()))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()

