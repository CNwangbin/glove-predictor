import numpy as np
import torch
import random
import os
import shutil
from collections import OrderedDict
import numpy as np
from torch.cuda.amp import autocast
from torch import distributed as dist

import csv
import os
from collections import OrderedDict

try:
    import wandb
except ImportError:
    pass

def print_trainable_parameters(model):
  trainable_params = 0
  all_param = 0
  for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
      trainable_params += param.numel()

  return (f"trainable params: {trainable_params} || all params: {all_param} "
          f"|| trainable: {(100 * trainable_params / all_param)}")

def get_outdir(path, *paths, inc=False):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir


def update_summary(epoch,
                   train_metrics,
                   eval_metrics,
                   filename,
                   write_header=True,
                   log_wandb=True):
    rowd = OrderedDict(epoch=epoch)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    if log_wandb:
        wandb.log(rowd)
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)

def save_checkpoint(checkpoint_state, epoch, is_best, checkpoint_dir):
    if (not torch.distributed.is_initialized()
        ) or torch.distributed.get_rank() == 0:
        if is_best:
            torch.save(checkpoint_state, os.path.join(checkpoint_dir, 'model_best.pth.tar'))

# def save_checkpoint(checkpoint_state, epoch, is_best, checkpoint_dir):
#     if (not torch.distributed.is_initialized()
#         ) or torch.distributed.get_rank() == 0:
#         # filename = 'checkpoint_' + str(epoch) + '.pth'
#         filename = 'checkpoint_last' + '.pth'
#         file_path = os.path.join(checkpoint_dir, filename)
#         torch.save(checkpoint_state, file_path)
#         if is_best:
#             shutil.copyfile(file_path,
#                             os.path.join(checkpoint_dir, 'model_best.pth.tar'))

def load_model_checkpoint(checkpoint_path):
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        # 将模型加载到CPU上
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model_state = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] if k.startswith('module') else k
                model_state[name] = v

            optimizer_state = checkpoint['optimizer']
        else:
            return None, None
        return model_state, optimizer_state
    else:
        raise FileNotFoundError('[!] No checkpoint found, start epoch 0')

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= (torch.distributed.get_world_size()
           if torch.distributed.is_initialized() else 1)
    return rt

def get_train_step(model, optimizer, scaler, gradient_accumulation_steps,
                   use_amp):
    def _step(inputs, optimizer_step=True):
        # Runs the forward pass with autocasting.
        with autocast(enabled=use_amp):
            outputs = model(**inputs)
            loss = outputs[0]
            loss /= gradient_accumulation_steps
            if torch.distributed.is_initialized():  # type: ignore
                reduced_loss = reduce_tensor(loss.data)
            else:
                reduced_loss = loss.data

        scaler.scale(loss).backward()

        if optimizer_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        torch.cuda.synchronize()
        return reduced_loss

    return _step


# 设置随机种子
def set_seed(seed_value=42):
    # 设置PyTorch的随机种子
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    # 设置NumPy的随机种子
    np.random.seed(seed_value)

    # 设置Python的随机种子
    random.seed(seed_value)

    # 使结果可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


