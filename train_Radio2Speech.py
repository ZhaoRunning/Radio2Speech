import logging
import time
import os
import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.utils.data.distributed
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from mel_Arguments import ArgParser
from RadioAudio_meldataset import Meldataset
from cnn_transformer.utils import AdamW, get_linear_schedule_with_warmup, LogSTFTMagnitudeLoss, checkpoint
from utils.mel_utils import AverageMeter, dist_average

# transformer + unet
from cnn_transformer.transunet import TransUnet as TransUnet

from evaluation import evaluate, evaluate_visual



def train_one_epoch(net, train_loader, optimizer, scheduler, criterion, epoch, accumulated_iter, tb_log, args):
    batch_time = AverageMeter()
    dataload_time = AverageMeter()
    total_loss = 0

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{args.epochs}', unit='batch') as pbar:
        torch.cuda.synchronize()
        tic = time.perf_counter()

        for i, batch_data in enumerate(train_loader):
            audio_amp, radio_amp = batch_data
            audio_amp = audio_amp.cuda()
            radio_amp = radio_amp.cuda()

            #measure data time
            torch.cuda.synchronize()
            dataload_time.update(time.perf_counter() - tic)

            audio_pred = net(radio_amp)

            # with torch.autograd.set_detect_anomaly(True):
            loss = criterion(audio_pred, audio_amp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # measure total time
            torch.cuda.synchronize()
            batch_time.update(time.perf_counter() - tic)
            tic = time.perf_counter()

            accumulated_iter += 1
            total_loss += loss.item()

            # display
            if i % args.disp_iter == 0:
                print('Epoch: [{}][{}/{}], batch time: {:.3f}, Data time: {:.3f}, loss: {:.4f}'
                  .format(epoch, i, len(train_loader),
                          batch_time.average(), dataload_time.average(), loss.item()))

            # add tensorboard
            tb_log.add_scalar('train/loss', loss.item(), accumulated_iter)
            tb_log.add_scalar('train/epoch', epoch, accumulated_iter)
            tb_log.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], accumulated_iter)

            # Manually update the progress bar, useful for streams such as reading files.
            pbar.update(1)
            if args.distributed:
                pbar.set_postfix(**{'loss (batch)': dist_average([total_loss / (i + 1)], i + 1)[0]})
            else:
                pbar.set_postfix(**{'loss (batch)': total_loss / (i+1)})

    return accumulated_iter


def train_net(args):
    # 0. initialize distribute and tensorboard
    args.gpu = 0
    args.world_size = 1
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    if not os.path.exists(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)
    tb_log = SummaryWriter(log_dir=args.tensorboard_dir)

    # 1. Create dataset
    train_sampler = None
    val_sampler = None
    train_set = Meldataset(args.list_train)
    val_set = Meldataset(args.list_val)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)

    # 2. Create data loaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                              shuffle=(train_sampler is None), sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, num_workers=4,
                            pin_memory=True, shuffle=False, sampler=val_sampler) #drop_last=False

    #3. Build the network: Transunet
    net = TransUnet(args.hidden_size, 
                    args.transformer_num_layers, 
                    args.mlp_dim, 
                    args.num_heads, 
                    args.transformer_dropout_rate, 
                    args.transformer_attention_dropout_rate
                    )
    net.load_from(weights=np.load(args.transunet_pretrained_path))

    # warp the network
    if args.distributed:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.cuda()
    # distribute
    if args.distributed:
        net = torch.nn.parallel.DistributedDataParallel(net,
                            device_ids=[args.local_rank], output_device=args.local_rank)

    # 4. Set up the optimizer, the criterion, and the learning rate scheduler.
    if args.opt == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)  # weight_decay=1e-5
    elif args.opt == 'adamw':  # add AdamW scheduler
        optimizer = AdamW(net.parameters(), args.learning_rate)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)

    if args.lr_scheduler == 'warmup':
        num_update_steps_per_epoch = len(train_loader)
        max_train_steps = args.epochs * num_update_steps_per_epoch
        warmup_steps = int(args.warmup_ratio * max_train_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, max_train_steps)
    else:
        step_size = args.step_size * len(train_loader)  # epoch * step_size per epoch
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.4)

    criterion = LogSTFTMagnitudeLoss()

    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.learning_rate}
    ''')

    # 5. load  checkpoint
    start_epoch = count_iter = 0
    # if args.local_rank == 0 and args.load_checkpoint is not None:
    if args.load_checkpoint is not None:
        dist.barrier()
        logging.info(f'Loading checkpoint net from: {args.load_checkpoint}')
        # map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
        package = torch.load(args.load_checkpoint, map_location='cpu')
        net_dict = net.state_dict()
        state_dict = {k: v for k, v in package.items() if k in net_dict.keys()}
        net_dict.update(state_dict)
        net.load_state_dict(net_dict)
    # initialize checkpoint package
    history = {'eval_loss': []}

    # 6. Begin training
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch) # syn different node in one epoch
        net.train()
        start = time.time()

        logging.info('-' * 70)
        logging.info('Training...')
        # train one epoch
        count_iter = train_one_epoch(net, train_loader, optimizer, scheduler, criterion, epoch, count_iter, tb_log, args)
        logging.info(f'Train Summary | End of Epoch {epoch} | Time {time.time() - start:.2f}s')

        # # Evaluation round
        logging.info('-' * 70)
        logging.info('Evaluating...')
        evaluate(net, val_loader, criterion, epoch, history, tb_log, count_iter)

        # evaluate samples every 'eval_every' argument number of epochs also evaluate on last epoch
        if (epoch + 1) % args.metrics_every == 0 or epoch == 0 or epoch == args.epochs - 1 :
            # Evaluate on the testset
            logging.info('-' * 70)
            logging.info('Calculating metrics...')
            evaluate_visual(net, val_loader, epoch, tb_log, count_iter, args)

        if args.local_rank == 0:
            #checkpointing
            checkpoint(net, history, epoch, optimizer, count_iter, args)


def main():
    parser = ArgParser()
    args = parser.parse_train_arguments()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    train_net(args)

if __name__ == '__main__':
    main()




