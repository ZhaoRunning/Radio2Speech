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

        # Transformer Unet setting
        parser.add_argument('--hidden_size', help='hidden size', type=int)
        parser.add_argument('--transformer_num_layers', help='transformer number of layers', type=int)
        parser.add_argument('--mlp_dim', help='transformer mlp dim', type=int)
        parser.add_argument('--num_heads', help='transformer number of heads', type=int)
        parser.add_argument('--transformer_dropout_rate', help='transformer dropout rate', type=float)
        parser.add_argument('--transformer_attention_dropout_rate', help='transformer attention dropout rate', type=float)
        parser.add_argument('--transformer_pretrained_path', help='path to the pre-trained transunet checkpoint', type=str)

        # visualize
        parser.add_argument('--disp_iter', dest='disp_iter', help='frequency to display the training information', type=int)

        #training
        parser.add_argument('--batch_size',help='train batch size', type=int) 
        parser.add_argument('--val_batch_size',help='val batch size', type=int)
        parser.add_argument('--learning_rate',help='learning rate for training', type=float) 
        parser.add_argument('--epochs',help='the number of training epochs', type=int) 
        
        # optimizer & scheduler
        parser.add_argument('--opt',help='optimizer', type=str)
        parser.add_argument('--lr_scheduler',help='scheduler', type=str)
        parser.add_argument('--warmup_ratio',help='warmup ratio', type=float)
        parser.add_argument('--step_size',help='step size epoch to adjust the learning rate', type=int) 

        # distributed
        parser.add_argument('--distributed', help='distributed training', default=True, type=str)
        parser.add_argument('--local_rank', help='distributed training for local_rank', default=0, type=int)

        # directory
        parser.add_argument('--list_train', help='list of training data', type=str)
        parser.add_argument('--list_val', help='list of evaluation data', type=str)

        # tensorboard
        parser.add_argument('--tensorboard_dir', help='output dir to save tensorboard file', type=str)

        # checkpoint
        parser.add_argument('--save_freq', help='frequency to save the checkpoint', type=int)
        parser.add_argument('--best_loss', default="inf", type=float)
        parser.add_argument('--load_checkpoint', help='load from pre-trained checkpoint', default=None, type=str)
        parser.add_argument('--save_ckpt', help='save model to checkpoint path', type=str)

        #evaluate
        parser.add_argument('--metrics_every', help='frequency to evaluate', type=int)

        self.parser = parser

    def parse_train_arguments(self):
        args = self.parser.parse_args()
        return args