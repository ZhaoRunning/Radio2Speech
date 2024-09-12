import argparse
import configargparse
from distutils.util import strtobool

class ArgParser(object):
    def __init__(self):
        # parser = argparse.ArgumentParser(description='Train a Unet')

        parser = configargparse.ArgumentParser(
            description="Train Radio2Speech to recover speech from radio signal",
            config_file_parser_class=configargparse.YAMLConfigFileParser,  # decide the config file syntax
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        )
        # general configuration
        parser.add("--config", is_config_file=True, help="config file path")  # add the config file

        # vit
        # For the position embedding, if this < 224, no need to change. LSpeech: 234, TIMIT: 224
        parser.add_argument('--img_size', type=int, default=224,
                            help='input patch size of network input')
        parser.add_argument('--num_classes', type=int,
                            default=1, help='output channel of network')
        parser.add_argument('--n_skip', type=int, default=3,
                            help='using number of skip-connect, default is num')
        parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16',
                            help='select one vit model')
        parser.add_argument('--vit_patches_size', type=int, default=8,
                            help='vit_patches_size, default is 16')
        parser.add_argument('--vit_pretrained_path', type=str, help='pretrained vit model path')

        # visualize
        parser.add_argument('--disp_iter', dest='disp_iter',
                            help='frequency to display the training information', default=50, type=int) #40

        #training
        parser.add_argument('--batch_size',
                            help='train batch size', default=60, type=int) #LJSpeech:256 TIMIT:128
        parser.add_argument('--val_batch_size',
                            help='val batch size', default=50, type=int)
        parser.add_argument('--learning_rate',
                            help='learning rate for training', default=0.004, type=float) #unet:0.004, transunet: 0.01
        parser.add_argument('--epochs',
                            help='the number of training epochs', default=420, type=int) # unet: 900, vit:450
        # optimizer & scheduler
        parser.add_argument('--opt', help='optimizer', default='adamw', type=str)
        parser.add_argument('--lr_scheduler', help='scheduler', default='warmup', type=str)
        parser.add_argument('--warmup_ratio', help='warmup ratio', default=0.1, type=float)
        parser.add_argument('--step_size',
                            help='step size epoch to adjust the learning rate', default=200, type=int) #350 vit:210

        # loss setting and parameter
        # parser.add_argument('--amp_wave_loss', dest='amp_wave_loss',
        #                     help='combine wave loss and wrapped stft amplitude loss', default=False)
        # parser.add_argument('--adjust_loss', dest='adjust_loss',
        #                     help='adjust the loss function at which epoch', default=45)
        # parser.add_argument('--factor_wave', dest='factor_wave', help='wave loss factor', default=0.5)
        # parser.add_argument('--ft_sc', dest='factor_sc',
        #                     help='Spectral Convergenge Loss factor', default=0.5)

        # distributed
        parser.add_argument('--distributed', help='distributed training', default=True, type=str)
        parser.add_argument('--local_rank', help='distributed training for local_rank',
                            default=0, type=int)

        # directory
        parser.add_argument('--list_train', help='list of training data', type=str)
        parser.add_argument('--list_val', help='list of evaluation data', type=str)

        # tensorboard
        parser.add_argument('--tensorboard_dir', help='output dir to save tensorboard file', type=str)

        # checkpoint
        # /home/zhaorn/checkpoint/compare/transunet_LJSpeech_channel128/net_best.pth
        parser.add_argument('--save_freq', help='frequency to save the checkpoint', type=int)
        parser.add_argument('--best_loss', default="inf", type=float)
        parser.add_argument('--load_checkpoint', help='load from pre-trained checkpoint',
                            default=None, type=str)
        parser.add_argument('--save_ckpt', help='save model to checkpoint path', type=str)

        #evaluate
        parser.add_argument('--metrics_every', help='frequency to evaluate', default=30, type=int) #30

        self.parser = parser

    def parse_train_arguments(self):
        args = self.parser.parse_args()
        return args