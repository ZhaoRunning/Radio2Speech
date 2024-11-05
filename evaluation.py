import logging

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import librosa
import librosa.display

import torch

from utils.mel_utils import dist_average

def melamp_visual(audio_amp, pred_amp, radio_amp):
    # record amplitude in tensorboard
    # audio_amp = 10**audio_amp
    # pred_amp = 10**pred_amp
    B = pred_amp.size(0)
    b = np.random.randint(0, B-1)
    amp_list = [audio_amp[b, 0, ...].cpu().numpy().transpose(1,0), radio_amp[b, 0, ...].cpu().numpy().transpose(1,0),
                pred_amp[b, 0, ...].cpu().numpy().transpose(1,0)]
    return amp_list

def evaluate(model, val_loader, criterion, epoch, history, tb_log, count_iter):
    logging.info(f'Evaluating at {epoch} epochs...')

    # initialize metrics
    # distribute training
    total_loss = 0
    total_count = 0

    # without distribute
    # loss_metrics = AverageMeter()

    model.eval()
    with torch.no_grad():
        # num_iters = len(val_loader)
        # prefetcher = data_prefetcher(val_loader)
        # audio_amp, _, radio_amp = prefetcher.next()
        # i = 0
        with tqdm(total=len(val_loader), desc='validation', unit='batch', leave=False) as pbar:
            for i, batch_data in enumerate(val_loader):
            # while radio_amp is not None:
            #
            #     i += 1
            #     if i > num_iters:
            #         break
                audio_amp, radio_amp = batch_data
                audio_amp = audio_amp.cuda()
                radio_amp = radio_amp.cuda()

                audio_pred_amp = model(radio_amp)

                eval_loss = criterion(audio_pred_amp, audio_amp)

                # distribute
                total_loss += eval_loss.item()
                total_count += audio_pred_amp.size(0)

                # without distribute
                # loss_metrics.update(eval_loss.item())

                pbar.update(1)

                # audio_amp, _,radio_amp = prefetcher.next()

                del eval_loss

    # distribute
    evaluation_loss = dist_average([total_loss / (i + 1)], i + 1)[0]
    history['eval_loss'].append(evaluation_loss)
    tb_log.add_scalar('eval/loss', evaluation_loss, count_iter)
    print('Evaluation Summary: Epoch: {}, Loss: {:.4f}'.format(epoch, evaluation_loss))

    # without distribute
    # history['eval_loss'].append(loss_metrics.average())
    # print('Evaluation Summary: Epoch: {}, Loss: {:.4f}'.format(epoch, loss_metrics.average()))


def evaluate_visual(model, val_loader, epoch, tb_log, count_iter, args):
    logging.info(f'Upload the mel spectrogram at {epoch} /epochs during evaluation')

    # initialize metrics
    tb_amp_list = []

    model.eval()
    with torch.no_grad():
        # num_iters = len(val_loader)
        #
        # prefetcher = data_prefetcher(val_loader)
        # audio_amp, audio_raw, radio_amp = prefetcher.next()
        # i = 0
        with tqdm(total=len(val_loader), desc='metric for val', unit='batch', leave=False) as pbar:
            # while radio_amp is not None:
            #
            #     i += 1
            #     if i > num_iters:
            #         break

            for i, batch_data in enumerate(val_loader):
                audio_amp, radio_amp = batch_data
                audio_amp = audio_amp.cuda()
                radio_amp = radio_amp.cuda()

                audio_pred_amp = model(radio_amp)

                # calculate the metrics and visualize
                amp_list = melamp_visual(audio_amp, audio_pred_amp, radio_amp)

                # distribute
                tb_amp_list.append(amp_list)

                pbar.update(1)
                # audio_amp, audio_raw, radio_amp = prefetcher.next()

    # add spectrogram in tensorboard
    b = np.random.randint(0, len(tb_amp_list) - 1)
    fig, axes = plt.subplots(3, 1, figsize=(6,6))
    for k, mag in enumerate(tb_amp_list[b]):
        axes[k].set_title(f"mean: {np.mean(mag):.3f}, "
                          f"std: {np.std(mag):.3f}, "
                          f"max: {np.max(mag):.3f}, "
                          f"min: {np.min(mag):.3f}")
        librosa.display.specshow(mag, x_axis='s', y_axis="mel", ax=axes[k], sr=8000, hop_length=128)
    plt.tight_layout()
    tb_log.add_figure(f'eval/{epoch}_melspect', fig, count_iter)


