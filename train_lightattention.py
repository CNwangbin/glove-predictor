import yaml
import argparse
import logging
import os
import wandb
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
import time
from collections import OrderedDict
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import torch.distributed as dist
import pytorch_warmup as warmup

from dataset import ImdbDataset
from models import LightAttentionPredictor
from utils import set_seed, get_train_step, save_checkpoint, load_model_checkpoint, update_summary, print_trainable_parameters
from metrics import AverageMeter, compute_metrics

import matplotlib.pyplot as plt
import math
import matplotlib.lines as mlines

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='esm-finetune model Training Config',
                                                 add_help=False)
parser.add_argument('-c',
                    '--config',
                    default='',
                    type=str,
                    metavar='FILE',
                    help='YAML config file specifying default arguments')
parser = argparse.ArgumentParser(
    description='Light attention predictor trainning config')
parser.add_argument('-dp',
                    '--data-path',
                    default='data/crc_cross_dataset_new',
                    type=str,
                    help='path of training data')
parser.add_argument('-de',
                    '--dim-embedding',
                    default=100,
                    type=int,
                    help='dimension of pretrained embedding')
parser.add_argument('-datten',
                    '--hidden-dim-attention',
                    default=50,
                    type=int,
                    help='hidden dimension of attention mlp layer')
parser.add_argument('-dembed',
                    '--hidden-dim-embedding',
                    default=50,
                    type=int,
                    help='hidden dimension of attention mlp layer')
parser.add_argument('-dpred',
                    '--hidden-dim-prediction',
                    default=50,
                    type=int,
                    help='hidden dimension of predicted mlp layer')
parser.add_argument('-act',
                    '--activation',
                    default='relu',
                    type=str,
                    choices=('relu', 'tanh', 'sigmoid'))
parser.add_argument('-drop',
                    '--dropout',
                    default='0.2',
                    type=str,
                    help='dropout rate in mlp')
parser.add_argument('-ep',
                    '--embedding-path',
                    default='data',
                    type=str,
                    help='path of embedding file dir')
parser.add_argument('-pe',
                    '--pretrained-embedding',
                    default='abundance-percentile_100',
                    type=str,
                    choices=('abundance-percentile_100',
                            'abundance-totalsum_100',
                            'braycurtis-percentile_100',
                            'braycurtis-totalsum_100',
                            'faith_100',
                            'jaccard_100',
                            'russell_rao_100',
                            'russell_rao_weight_100',
                            'phylogeny_100',
                            'PCA_100'
                            ))
parser.add_argument('-nd',
                    '--num-dataset',
                    default=7,
                    type=int,
                    help='how many dataset to use for cross validation')
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs',
                    default=100,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-j',
                    '--workers',
                    type=int,
                    default=4,
                    metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('-b',
                    '--batch-size',
                    default=32,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 32) per gpu')
parser.add_argument('-lr',
                    '--learning-rate',
                    default='3e-5',
                    type=str,
                    help='initial learning rate')
parser.add_argument('-mlr',
                    '--min-learning-rate',
                    default=1e-6,
                    type=float,
                    help='minimal learning rate',)
parser.add_argument(
    '--lr-schedule',
    default='cosine',
    type=str,
    metavar='SCHEDULE',
    choices=[None, 'cosine', 'exponential'],
    help='Type of LR schedule: {}, {}'.format(None, 'exponential'))
parser.add_argument('--optimizer',
                    default='adamw',
                    type=str,
                    choices=('sgd', 'adam', 'adamw'))
parser.add_argument('-wd',
                    '--weight-decay',
                    default=0.01,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 0.0)',
                    dest='weight_decay')
parser.add_argument('--warmup_period',
                    default=20,
                    type=int,
                    help='warmup period, if > 0, use linear warmup')
parser.add_argument(
    '--no-checkpoints',
    action='store_false',
    dest='save_checkpoints',
    help='do not store any checkpoints, useful for benchmarking',
)
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--output-dir',
                    default='./work_dirs',
                    type=str,
                    help='output directory for model and log')
parser.add_argument('--log-wandb',
                    action='store_true',
                    help='while to use wandb log systerm')
parser.add_argument('--experiment', default='glove-embedding-prediction', type=str)
parser.add_argument('--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--training-only',
                    action='store_true',
                    help='do not evaluate')
parser.add_argument(
    '--amp',
    action='store_true',
    default=False,
    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--native-amp',
                    action='store_true',
                    default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument(
    '--early-stopping-patience',
    default=10,
    type=int,
    metavar='N',
    help='early stopping after N epochs without improving',
)
parser.add_argument(
    '--gradient_accumulation_steps',
    default=1,
    type=int,
    metavar='N',
    help='=To run gradient descent after N steps',
)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument(
    '--static-loss-scale',
    type=float,
    default=1,
    help='Static loss scale',
)
parser.add_argument(
    '--dynamic-loss-scale',
    action='store_true',
    help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
    '--static-loss-scale.',
)

def main(args):
    args.save_checkpoints=False
    args.log_wandb=False
    # print('log to wandb')
    # wandb.init(project=args.experiment, config=args, entity='cnwangbi')
    set_seed(args.seed)

    logger.info('light attention predictor cross dataset LODO training')

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.local_rank = 0

    args.gpu = 0
    args.world_size = 1
    args.rank = 0

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()

        logger.info(
            'Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
            % (args.rank, args.world_size))
    else:
        logger.info('Training with a single process on %s .' % args.gpu)

    with open('data/microbes_intersection.txt', 'r') as f:
        microbes = f.read().splitlines()
    if args.activation == 'relu':
        activation = nn.ReLU()
    elif args.activation == 'sigmoid':
        activation = nn.Sigmoid()
    elif args.activation == 'tanh':
        activation = nn.Tanh()
    else:
        raise ValueError("Invalid activation function")
    study_metrics = []
    for i in range(1,args.num_dataset+1):
        logger.info('validate on dataset %d', i)
        train_dataset = ImdbDataset(biom_table=os.path.join(args.data_path, f'train_{i}.biom'),
                                    metadata=os.path.join(args.data_path, 'metadata.txt'),
                                    group='group',
                                    microbes=microbes)
        test_dataset = ImdbDataset(biom_table=os.path.join(args.data_path, f'test_{i}.biom'),
                                    metadata=os.path.join(args.data_path, 'metadata.txt'),
                                    group='group',
                                    microbes=microbes)

        train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.workers,
                                pin_memory=True,
                                collate_fn=train_dataset.collate_fn)
        test_loader = DataLoader(test_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.workers,
                                pin_memory=True,
                                collate_fn=test_dataset.collate_fn)
        fid_dict = train_dataset()
        model = LightAttentionPredictor(
            fid_dict=fid_dict,
            d_model=args.dim_embedding,
            d_mlp_atten=args.hidden_dim_attention,
            d_mlp_embed=args.hidden_dim_attention,
            d_mlp_pred=args.hidden_dim_prediction,
            activation=activation,
            p_drop=args.dropout,
            classification=True,
            glove_embedding_path=os.path.join(args.embedding_path, args.pretrained_embedding + '.txt'),
            embedding_freeze=True
        )
        model.cuda()
        # optimizer and lr policy
        if args.optimizer == 'adamw':
            optimizer = optim.AdamW(filter(
                lambda p: p.requires_grad,
                model.parameters(),
            ),
                lr=args.learning_rate,
                weight_decay=args.weight_decay)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(filter(
                lambda p: p.requires_grad,
                model.parameters(),
            ),
                lr=args.learning_rate,
                weight_decay=args.weight_decay)
        else:
            print(f'optimizer {args.optimizer} is not implemented.')

        if args.lr_schedule == None:
            lr_schedule = None
        elif args.lr_schedule == 'cosine':
            lr_schedule = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=args.min_learning_rate)
        elif args.lr_schedule == 'exponential':
            lr_schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
        else:
            print(f'lr_schedule {args.lr_schedule} is not implemented.')

        if args.warmup_period > 0:
            warmup_period = args.warmup_period
            warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period)
        else:
            warmup_scheduler = None

        model.cuda()

        if args.resume is not None:
            if args.local_rank == 0:
                model_state, _ = load_model_checkpoint(args.resume)
                model.load_state_dict(model_state)

        scaler = torch.cuda.amp.GradScaler(
            init_scale=args.static_loss_scale,
            growth_factor=2,
            backoff_factor=0.5,
            growth_interval=100 if args.dynamic_loss_scale else 1000000000,
            enabled=args.amp,
        )

        if args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[args.gpu],
                    output_device=args.gpu,
                    find_unused_parameters=True)
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(
                    model, output_device=0, find_unused_parameters=True)
        else:
            model.cuda()

        start_epoch = 0
        if args.start_epoch is not None:
            # a specified start_epoch will always override the resume epoch
            start_epoch = args.start_epoch
        if lr_schedule is not None and start_epoch > 0:
            lr_schedule.step(start_epoch)

        if args.local_rank == 0:
            logger.info('Scheduled epochs: {}'.format(args.epochs))

        gradient_accumulation_steps = args.gradient_accumulation_steps
        train_metric_list, val_metric_list = train_loop(model,
                                                optimizer,
                                                lr_schedule,
                                                scaler,
                                                gradient_accumulation_steps,
                                                train_loader,
                                                test_loader,
                                                use_amp=args.amp,
                                                logger=logger,
                                                start_epoch=start_epoch,
                                                end_epoch=args.epochs,
                                                early_stopping_patience=args.early_stopping_patience,
                                                skip_training=args.evaluate,
                                                skip_validation=args.training_only,
                                                save_checkpoints=args.save_checkpoints,
                                                output_dir=args.output_dir,
                                                log_wandb=False,
                                                log_interval=args.log_interval,
                                                warmup_scheduler=warmup_scheduler,
                                                warmup_period=warmup_period)
        train_losses = [x['loss'] for x in train_metric_list]
        val_losses = [x['loss'] for x in val_metric_list]
        val_aucs = [x['auc'] for x in val_metric_list]
        study_metrics.append({'train_losses': train_losses, 'val_losses': val_losses, 'val_aucs': val_aucs})
        # test_metrics = evaluate(model, test_loader, args.amp, logger, args.log_interval)
        best_test_metrics = max(val_metric_list, key=lambda x: x['auc'])
        with open(sweep_log_path, 'a') as f:
            f.write(f'Dataset {i} best test metrics:' + str(best_test_metrics.values()) + '\n')
        logger.info('Test: %s' % (best_test_metrics))
        logger.info('Experiment ended.')

    # plot the training and validation loss
    plot_metrics(study_metrics)

def plot_metrics(study_metrics):
    n = len(study_metrics)
    rows = 2
    cols = math.ceil(n / rows)
    fig, axs = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))

    if n == 1:
        axs = [[axs]]
    elif rows == 1:
        axs = [axs]
    elif cols == 1:
        axs = [[ax] for ax in axs]

    for i, metrics in enumerate(study_metrics):
        row = i // cols
        col = i % cols
        ax = axs[row][col]
        
        epochs = range(1, len(metrics['train_losses']) + 1)
        
        ax.plot(epochs, metrics['train_losses'], label='Train loss', color='blue')
        ax.plot(epochs, metrics['val_losses'], label='Valid loss', color='green')
        ax.plot(epochs, metrics['val_aucs'], label='Valid AUC', color='orange')
        
        ax.set_title(f'Dataset {i+1}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('loss/AUC')

    # Create legend handles manually
    handles = [mlines.Line2D([], [], color='blue', label='Train loss'),
            mlines.Line2D([], [], color='green', label='Valid loss'),
            mlines.Line2D([], [], color='orange', label='Valid AUC')]

    # Add a new subplot for the legend
    ax_legend = fig.add_subplot(111)
    ax_legend.axis('off')
    ax_legend.legend(handles=handles, loc='upper right')

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(args.output_dir, 'metrics.png'))

def train(model,
          loader,
          optimizer,
          scaler,
          gradient_accumulation_steps,
          use_amp,
          epoch,
          logger,
          log_interval=1,
          warmup_scheduler=None,
          warmup_period=0,
          lr_scheduler=None):
    batch_time_m = AverageMeter('Time', ':6.3f')
    data_time_m = AverageMeter('Data', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')

    step = get_train_step(model, optimizer, scaler,
                          gradient_accumulation_steps, use_amp)

    model.train()
    optimizer.zero_grad()
    steps_per_epoch = len(loader)
    end = time.time()
    for idx, batch in enumerate(loader):
        # Add batch to GPU
        batch = {key: val.cuda() for key, val in batch.items()}
        data_time = time.time() - end
        optimizer_step = ((idx + 1) % gradient_accumulation_steps) == 0
        loss = step(batch, optimizer_step)
        batch_size = batch['labels'].shape[0]

        it_time = time.time() - end
        batch_time_m.update(it_time)
        data_time_m.update(data_time)
        losses_m.update(loss.item(), batch_size)

        if lr_scheduler is not None:
            if warmup_scheduler is not None:
                with warmup_scheduler.dampening():
                    if warmup_scheduler.last_step + 1 >= warmup_period:
                        lr_scheduler.step()
        end = time.time()
        if (idx % log_interval == 0) or (idx == steps_per_epoch - 1):
            if not torch.distributed.is_initialized(
            ) or torch.distributed.get_rank() == 0:
                learning_rate = optimizer.param_groups[0]['lr']
                log_name = 'Train-log'
                logger.info(
                    '{0}: [epoch:{1:>2d}] [{2:>2d}/{3}] '
                    'DataTime: {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'BatchTime: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f}) '
                    'lr: {lr:>4.6f} '.format(log_name,
                                             epoch + 1,
                                             idx,
                                             steps_per_epoch,
                                             data_time=data_time_m,
                                             batch_time=batch_time_m,
                                             loss=losses_m,
                                             lr=learning_rate))
    return OrderedDict([('loss', losses_m.avg)])


def evaluate(model, loader, use_amp, logger, log_interval=10):
    batch_time_m = AverageMeter('Time', ':6.3f')
    data_time_m = AverageMeter('Data', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')

    model.eval()
    steps_per_epoch = len(loader)
    end = time.time()
    # Variables to gather full output
    true_labels= []
    pred_labels = []
    for idx, batch in enumerate(loader):
        batch = {key: val.cuda() for key, val in batch.items()}
        labels = batch['labels']
        labels = labels.float()
        data_time = time.time() - end
        with torch.no_grad(), autocast(enabled=use_amp):
            outputs = model(**batch)
            loss = outputs[0]
            logits = outputs[1]
            preds = torch.sigmoid(logits)
            preds = preds.detach().cpu()

        torch.cuda.synchronize()

        preds = preds.numpy()
        labels = labels.to('cpu').numpy()
        true_labels.append(labels)
        pred_labels.append(preds)

        batch_size = labels.shape[0]
        data_time = time.time() - end
        it_time = time.time() - end
        end = time.time()

        batch_time_m.update(it_time)
        data_time_m.update(data_time)
        losses_m.update(loss.item(), batch_size)
        if (idx % log_interval == 0) or (idx == steps_per_epoch - 1):
            if not torch.distributed.is_initialized(
            ) or torch.distributed.get_rank() == 0:
                logger_name = 'Test-log'
                logger.info(
                    '{0}: [{1:>2d}/{2}] '
                    'DataTime: {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f}) '.format(
                        logger_name,
                        idx,
                        steps_per_epoch,
                        data_time=data_time_m,
                        batch_time=batch_time_m,
                        loss=losses_m))
    # Flatten outputs
    true_labels = np.concatenate(true_labels, axis=0)
    pred_labels = np.concatenate(pred_labels, axis=0)
    # avg_auc
    metric_dict = compute_metrics(true_labels, pred_labels)
    # {'acc': acc, 'mcc': mcc, 'roc_auc': roc_auc, 'aupr': aupr, 'f1':f1, 'spec': specificity, 'sens': sensitivity, 'p': precision, 'r': recall}
    metrics = OrderedDict([('loss', losses_m.avg), 
                           ('acc', metric_dict['acc']), 
                           ('mcc', metric_dict['mcc']), 
                           ('auc', metric_dict['roc_auc']), 
                           ('aupr', metric_dict['aupr']), 
                           ('f1', metric_dict['f1']),
                           ('spec', metric_dict['spec']),
                           ('sens', metric_dict['sens']),
                           ('p', metric_dict['p']),
                           ('r', metric_dict['r'])])
    return metrics

def train_loop(model,
               optimizer,
               lr_scheduler,
               scaler,
               gradient_accumulation_steps,
               train_loader,
               val_loader,
               use_amp,
               logger,
               start_epoch=0,
               end_epoch=0,
               early_stopping_patience=-1,
               skip_training=False,
               skip_validation=False,
               save_checkpoints=True,
               output_dir='./',
               log_wandb=True,
               log_interval=10,
               warmup_scheduler=None,
               warmup_period=0):
    if early_stopping_patience > 0:
        epochs_since_improvement = 0

    best_metric = 0
    train_metric_list = []
    val_metric_list = []
    logger.info('Evaluate validation set before start training')
    eval_metrics = evaluate(model, val_loader, use_amp, logger, log_interval)
    logger.info('Evaluation: %s' % (eval_metrics))
    logger.info(f'RUNNING EPOCHS FROM {start_epoch} TO {end_epoch}')
    for epoch in range(start_epoch, end_epoch):
        is_best = False
        if not skip_training:
            train_metrics = train(model, train_loader, optimizer, scaler,
                                  gradient_accumulation_steps, use_amp, epoch,
                                  logger, log_interval, warmup_scheduler, warmup_period=warmup_period, lr_scheduler=lr_scheduler)

            logger.info('[Epoch %d] training: %s' % (epoch + 1, train_metrics))

        if not skip_validation:
            eval_metrics = evaluate(model, val_loader, use_amp, logger,
                                    log_interval)

            logger.info('[Epoch %d] Evaluation: %s' %
                        (epoch + 1, eval_metrics))
            train_metric_list.append(train_metrics)
            val_metric_list.append(eval_metrics)
        if eval_metrics['auc'] > best_metric:
            is_best = True
            best_metric = eval_metrics['auc']
        if log_wandb and (not torch.distributed.is_initialized()
                          or torch.distributed.get_rank() == 0):
            update_summary(epoch,
                           train_metrics,
                           eval_metrics,
                           os.path.join(output_dir, 'summary.csv'),
                           write_header=best_metric is None,
                           log_wandb=log_wandb)

        if save_checkpoints and (not torch.distributed.is_initialized()
                                 or torch.distributed.get_rank() == 0):
            checkpoint_state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_metric': eval_metrics['acc'],
                'optimizer': optimizer.state_dict(),
            }
            logger.info('[*] Saving model epoch %d...' % (epoch + 1))
            save_checkpoint(checkpoint_state,
                            epoch,
                            is_best,
                            checkpoint_dir=output_dir)

        if early_stopping_patience > 0:
            if not is_best:
                epochs_since_improvement += 1
            else:
                epochs_since_improvement = 0
            if epochs_since_improvement >= early_stopping_patience:
                break
    return train_metric_list, val_metric_list


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


if __name__ == '__main__':
    args, args_text = _parse_args()
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    task_name = args.data_path.split('/')[-1].split('_')[0]
    pretrained_name = args.pretrained_embedding
    full_name = task_name + '_' + pretrained_name + '_' + f'hda{args.hidden_dim_attention}' + '_' + f'hdp{args.hidden_dim_prediction}'  + '_' + f'hde{args.hidden_dim_embedding}' + '_' + f"act{args.activation}" + '_' + f'drop{args.dropout}' + '_' + f'lr{args.learning_rate}' + '_' + f'bs{args.batch_size}'
    args.learning_rate = float(args.learning_rate)
    args.dropout = float(args.dropout)
    global sweep_log_path
    sweep_log_path = os.path.join(args.output_dir, 'sweep.log')
    with open(sweep_log_path, 'a') as f:
        f.write(full_name + '\n')
    args.output_dir = os.path.join(args.output_dir, full_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)

    logger = logging.getLogger('')
    filehandler = logging.FileHandler(
        os.path.join(args.output_dir, 'summary.log'))
    streamhandler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    main(args)