# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
import logging
from typing import Optional, Literal
from tqdm import tqdm
import random

# import torch.nn as nn
from torch.autograd import Variable

import numpy as np

import argparse


def str2bool(v):
    if v.lower() in ("true", "yes", "t", "y"):
        return True
    elif v.lower() in ("false", "no", "f", "n"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.2f}".format(name, float(meter)) if isinstance(meter, (int, float)) else "{}: {}".format(name, meter)
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        logger = logging.getLogger("MetricLogger")

        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    # print(log_msg.format(
                    #     i, len(iterable), eta=eta_string,
                    #     meters=str(self),
                    #     time=str(iter_time), data=str(data_time),
                    #     memory=torch.cuda.max_memory_allocated() / MB))
                    logger.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    # print(log_msg.format(
                    #     i, len(iterable), eta=eta_string,
                    #     meters=str(self),
                    #     time=str(iter_time), data=str(data_time)))
                    logger.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))

            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # print('{} Total time: {} ({:.4f} s / it)'.format(
        #     header, total_time_str, total_time / len(iterable)))
        logger.info('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


# def get_rank():
#     if not is_dist_avail_and_initialized():
#         return 0
#     return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    # if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    #     args.rank = int(os.environ["RANK"])
    #     args.world_size = int(os.environ['WORLD_SIZE'])
    #     args.gpu = int(os.environ['LOCAL_RANK'])
    # elif 'SLURM_PROCID' in os.environ:
    #     args.rank = int(os.environ['SLURM_PROCID'])
    #     args.gpu = args.rank % torch.cuda.device_count()
    # else:
    print('Not using distributed mode')
    args.distributed = False
    return

    # args.distributed = True

    # torch.cuda.set_device(args.gpu)
    # args.dist_backend = 'nccl'
    # print('| distributed init (rank {}): {}'.format(
    #     args.rank, args.dist_url), flush=True)
    # torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                                      world_size=args.world_size, rank=args.rank)
    # torch.distributed.barrier()
    # setup_for_distributed(args.rank == 0)


def get_logger(
    level: int,
    logger_fp: str,
    name: Optional[str] = None,
    mode: str = "w",
    format: str = "%(asctime)s - %(funcName)s - %(levelname)s - %(message)s"
):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    file_handler = logging.FileHandler(logger_fp, "w")
    file_handler.setLevel(level)
    formatter = logging.Formatter(format)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


class SizeEstimator(object):

    def __init__(self, model, input_size=(1,1,32,32), bits=32):
        '''
        Estimates the size of PyTorch models in memory
        for a given input size
        '''
        self.model = model
        self.input_size = input_size
        self.bits = bits
        return

    def get_parameter_sizes(self):
        '''Get sizes of all parameters in `model`'''
        mods = list(self.model.modules())
        sizes = []
        
        for i in range(1,len(mods)):
            m = mods[i]
            p = list(m.parameters())
            for j in range(len(p)):
                sizes.append(np.array(p[j].size()))

        self.param_sizes = sizes
        return

    def get_output_sizes(self):
        '''Run sample input through each layer to get output sizes'''
        input_ = Variable(torch.FloatTensor(*self.input_size), volatile=True)
        mods = list(self.model.modules())
        out_sizes = []
        for i in range(1, len(mods)):
            m = mods[i]
            out = m(input_)
            out_sizes.append(np.array(out.size()))
            input_ = out

        self.out_sizes = out_sizes
        return

    def calc_param_bits(self):
        '''Calculate total number of bits to store `model` parameters'''
        total_bits = 0
        for i in range(len(self.param_sizes)):
            s = self.param_sizes[i]
            bits = np.prod(np.array(s))*self.bits
            total_bits += bits
        self.param_bits = total_bits
        return

    def calc_forward_backward_bits(self):
        '''Calculate bits to store forward and backward pass'''
        total_bits = 0
        for i in range(len(self.out_sizes)):
            s = self.out_sizes[i]
            bits = np.prod(np.array(s))*self.bits
            total_bits += bits
        # multiply by 2 for both forward AND backward
        self.forward_backward_bits = (total_bits*2)
        return

    def calc_input_bits(self):
        '''Calculate bits to store input'''
        self.input_bits = np.prod(np.array(self.input_size))*self.bits
        return

    def estimate_size(self):
        '''Estimate model size in memory in megabytes and bits'''
        self.get_parameter_sizes()
        self.get_output_sizes()
        self.calc_param_bits()
        self.calc_forward_backward_bits()
        self.calc_input_bits()
        total = self.param_bits + self.forward_backward_bits + self.input_bits

        total_megabytes = (total/8)/(1024**2)
        return total_megabytes, total


def repr_set(s: set):
    t = r"\{"
    for i in s:
        t += r"{}, ".format(i)
    t += r"\}"
    return t


# from sklearn.tree._export import _BaseTreeExporter
from numbers import Integral
from io import StringIO


def _color_brew(n):
    """Generate n colors with equally spaced hues.

    Parameters
    ----------
    n : int
        The number of colors required.

    Returns
    -------
    color_list : list, length n
        List of n tuples of form (R, G, B) being the components of each color.
    """
    color_list = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in np.arange(25, 385, 360.0 / n).astype(int):
        # Calculate some intermediate values
        h_bar = h / 60.0
        x = c * (1 - abs((h_bar % 2) - 1))
        # Initialize RGB with same hue & chroma as our color
        rgb = [
            (c, x, 0),
            (x, c, 0),
            (0, c, x),
            (0, x, c),
            (x, 0, c),
            (c, 0, x),
            (c, x, 0),
        ]
        r, g, b = rgb[int(h_bar)]
        # Shift the initial RGB values to match value and store
        rgb = [(int(255 * (r + m))), (int(255 * (g + m))), (int(255 * (b + m)))]
        color_list.append(rgb)

    return color_list


class Sentinel:
    def __repr__(self):
        return '"tree.dot"'


SENTINEL = Sentinel()

# Ours
def sample_mask(logits, tau=0.3):
    shape = logits.shape
    mask = gumbel_sigmoid(logits, tau=tau)
    return mask

def gumbel_sigmoid(logits, tau=1):
    dims = logits.dim()
    logistic_noise = sample_logistic(logits.size(), out=logits.data.new())
    y = logits + logistic_noise
    return torch.sigmoid(y / tau)

def sample_logistic(shape, out=None):
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return torch.log(U) - torch.log(1-U)

def get_mask(logits, tau=0.3):
    mask = sample_mask(logits, tau)
    return mask

class supervised_contrastive_loss(nn.Module):
    def __init__(self, args):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning :
        https://arxiv.org/abs/2004.11362

        :param temperature: int
        """
        super(supervised_contrastive_loss, self).__init__()
        self.temperature = args.temperature

    def forward(self, projections, targets):
        """
        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")

        dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / (cardinality_per_samples + 1e-5)
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss
    
def process_dataset(model, data_loader_train, batch_size=32, device='cuda'):
    model.eval()  # Set model to evaluation mode

    results = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation for inference
        for data, labels, _, ind in data_loader_train:
            data = data.to(device)
            labels = labels.to(device)

            # Forward pass
            _cls_tokens, patch_tokens, frz_cls_tokens, frz_patch_tokens = model.features.forward_all(data)
            
            cls_tokens = model.add_on_layers(_cls_tokens)
            
            results.append(cls_tokens.cpu())  # Move results to CPU
            all_labels.append(labels.cpu())  # Move labels to CPU

    # Concatenate all results and labels
    zs = torch.cat(results, dim=0)  # Shape: (n_samples, dim)
    y = torch.cat(all_labels, dim=0)  # Shape: (n_samples,)
    
    syn_zs, syn_y = synthesis_novel(zs, y, method='category', swap_method='a', sampling_method='high', n_sample_classes=10000, k_top=256) 
    return syn_zs, syn_y    
    
    
def synthesis_novel(features,
                    labels,
                    method: Literal['instance', 'category'] = 'instance',
                    swap_method: Literal['a', 'b'] = 'a',
                    sampling_method: Literal['high', 'low', 'random'] = 'random',
                    n_sample_classes: int = 10000,
                    k_top:int =200):
    ### read data from files
    num_classes = len(np.unique(labels))
    num_samples = len(features)
    
    ori_features = features.clone().detach()
    mu_f = [features[labels==c].mean(0) for c in range(num_classes)]
    std_f = [features[labels==c].std(0) for c in range(num_classes)]
    
    topk_ind = torch.stack([mu.topk(k=k_top).indices for mu in mu_f], dim=0)
    topk_val = torch.stack([mu.topk(k=k_top).values for mu in mu_f], dim=0)

    #region compute topk overlapping
    num_classes, K = topk_ind.shape
    overlap_ratio = torch.zeros((num_classes, num_classes))
    
    for i in range(num_classes):
        for j in range(num_classes):
            intersection = len(set(topk_ind[i].tolist()) & set(topk_ind[j].tolist()))
            overlap_ratio[i, j] = intersection / K

    overlap_ratio.fill_diagonal_(0)
    conf_ind_top = overlap_ratio.topk(3).indices
    conf_ind_low = (-overlap_ratio).topk(3).indices
    #endregion
        
    mixed_features = torch.zeros([ori_features.size(0), num_classes, ori_features.size(1)]) ### output mixing feature: samples (class A) x classes B x dimensions
    
    for i in tqdm(range(ori_features.size(0))):
        line = ori_features[i].clone()
        class_a = labels[i]
        for class_b in range(num_classes):
            if method == 'instance':
                class_mask = torch.where(labels == class_b)
                sample_b = random.choice(class_mask[0]).item()
                feat_b = ori_features[sample_b]
            elif method == 'category':
                feat_b = mu_f[class_b]
            else:
                raise NotImplementedError()
            replace_ind = topk_ind[class_a if swap_method == 'a' else class_b]
            line[replace_ind] = feat_b[replace_ind]
            mixed_features[i, class_b] = line

    random_classes = np.random.choice(np.unique(labels), size=min(n_sample_classes, num_classes), replace=False)
    mask = np.isin(labels, random_classes)
    filtered_features = mixed_features[mask]
    filtered_labels = labels[mask]
    filtered_conf_ind_top = conf_ind_top[filtered_labels]
    filtered_conf_ind_low = conf_ind_low[filtered_labels]
    
    if sampling_method == 'high':
        ### design 1: high topk overlapping
        select_mixing_features = \
            filtered_features[
            torch.arange(filtered_features.size(0)), 
            filtered_conf_ind_top[:, 0]]
    elif sampling_method == 'low':
        ### design 2: low topk overlapping
        select_mixing_features = \
            filtered_features[
            torch.arange(filtered_features.size(0)), 
            filtered_conf_ind_low[:, 1]]
    elif sampling_method == 'random':
        ### design 3: random
        random_mixing_labels = torch.zeros_like(labels)
        for c in labels.unique():
            class_mask = torch.where(labels==c)
            while 1:
                rand_class = torch.randint(low=0, high=num_classes, size=[1])
                if rand_class != c:
                    break
            random_mixing_labels[class_mask] = rand_class.item()
        select_mixing_features = \
            filtered_features[
            torch.arange(filtered_features.size(0)), 
            random_mixing_labels[mask]]

    # features_2d = visualize_reduction([
    #     filtered_features[:, 0], select_mixing_features], labels=filtered_labels, method='tsne')
    return select_mixing_features, filtered_labels
    