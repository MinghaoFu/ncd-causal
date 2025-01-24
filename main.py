import time
import torch
import random
import argparse
import datetime
import logging
import numpy as np

from pathlib import Path
from timm.utils import NativeScaler, ModelEma
from phe_model import construct_PPNet_dino
from train_eval import train_and_evaluate, evaluate, evaluate_train_dataset
from data.datasets import build_dataset
from utils import create_optimizer, create_scheduler, get_logger, load_checkpoint_for_ema, str2bool, get_dis_max

from config import pretrain_path, oxford_pet_root, cub_root, car_root, food_101_root, inaturalist_root

import os
import wandb  # Add wandb import
os.environ["WANDB_SILENT"] = "true"

def get_args_parser():

    parser = argparse.ArgumentParser('PHE training', add_help=False)

    # training parameters
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--prototype_shape', nargs='+', type=int, default=[2000, 192, 1, 1])
    parser.add_argument('--prototype_activation_function', type=str, default='log')
    parser.add_argument('--add_on_layers_type', type=str, default='regular')

    parser.add_argument('--use_global', type=str2bool, default=True)
    parser.add_argument('--global_proto_per_class', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--save_ep_freq', default=10, type=int, help='save epoch frequency')

    parser.add_argument('--hash_code_length', default=12, type=int)
    parser.add_argument('--prototype_dim', default=768, type=int)
    parser.add_argument('--alpha', default=0.1, type=float, help='loss weight alpha')
    parser.add_argument('--beta', default=3.0, type=float, help='loss weight beta')

    # Model Exponential Moving Average
    parser.add_argument('--model_ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER', help='LR scheduler (default: "cosine"')
    parser.add_argument('--features_lr', type=float, default=1e-4)
    parser.add_argument('--add_on_layers_lr', type=float, default=1e-3)
    parser.add_argument('--prototype_vectors_lr', type=float, default=1e-3)
    parser.add_argument('--decoder_lr', type=float, default=1e-3)
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-4, metavar='LR', help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=10, metavar='N', help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')


    # Dataset parameters
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--mask_theta', type=float, default=0.1)
    parser.add_argument('--labeled_nums', type=int, default=0)
    parser.add_argument('--unlabeled_nums', type=int, default=0)
    parser.add_argument('--data_set', default='cub', 
    choices=['cub', 'scars', 'food', 'pets', 'Actinopterygii', 'Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Chromista', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Protozoa', 'Reptilia'],
                        type=str, help='Image Net dataset path')

    parser.add_argument('--output_dir', default='exp/', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=1028, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)

    # Ours
    parser.add_argument('--y_embed_dim', default=128, type=int, help='')
    # zs = 768, zc = 128
    parser.add_argument('--zc_dim', default=128, type=int, help='dimension of invariant latent semantics')
    parser.add_argument('--zs_dim', default=768, type=int, help='dimension of changing latent semantics')
    parser.add_argument('--ttt', action='store_true', help='')
    parser.add_argument('--l_recon', default=1e-4, type=float, help='')
    parser.add_argument('--l_spa', default=1e-4, type=float, help='')
    parser.add_argument('--l_ind', default=1e-4, type=float, help='')

    parser.add_argument('--intra', action='store_true', help='data augmentation or not for intra-class loss')
    parser.add_argument('--n_intra', default=1, type=int, help='n aug')
    parser.add_argument('--l_intra', default=1e-2, type=float, help='lambda for intra-class loss')
    parser.add_argument('--temperature', default=1, type=float, help='')
    
    parser.add_argument('--syn', action='store_true', help='')

    parser.add_argument('--recon_attn', action='store_true', help='')
    parser.add_argument('--recon_cls', action='store_true', help='')
    parser.add_argument('--n_recon_epoch', default=100, type=int, help='')  
    parser.add_argument('--eval_train', action='store_true', help='Perform evaluation only on training data')
    
    # lr reconstruction, flow
    # flow
    # remove zc
    # use mlp d = f(y,c)  

    
    return parser


def get_outlog(args):

    if args.eval: # evaluation only
        logfile_dir = os.path.join(args.output_dir, "eval-logs")
    else: # training
        logfile_dir = os.path.join(args.output_dir, "train-logs")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    tb_dir = os.path.join(args.output_dir, "tf-logs")
    tb_log_dir = os.path.join(tb_dir, args.data_set)
    os.makedirs(logfile_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    logger = get_logger(
        level=logging.INFO,
        mode="w",
        name=None,
        logger_fp=os.path.join(
            logfile_dir,
            args.data_set + ".log"
        )
    )

    return logger


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    # fix the seed for reproducibility
    set_seed(args.seed)

    # tb_writer, logger = get_outlog(args)
    logger = get_outlog(args)

    logger.info("Start running with args: \n{}".format(args))

    device = torch.device(args.device)

    dataset_train, dataset_val, test_dataset_unlabelled = build_dataset(args=args)

    logger.info("train {} test: {}".format(len(dataset_train), len(dataset_val)))
    logger.info("test_dataset_unlabelled: {}".format(len(test_dataset_unlabelled)))

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    test_loader_unlabelled = torch.utils.data.DataLoader(
        test_dataset_unlabelled, 
        num_workers=8,
        batch_size=256, 
        shuffle=False, 
        pin_memory=False)
    
    args.prototype_shape=[args.labeled_nums * args.global_proto_per_class, args.prototype_dim, 1, 1]

    model = construct_PPNet_dino(args, img_size=args.img_size,
                                prototype_shape=args.prototype_shape,
                                num_classes=args.labeled_nums,
                                use_global=args.use_global,
                                global_proto_per_class=args.global_proto_per_class,
                                prototype_activation_function=args.prototype_activation_function,
                                add_on_layers_type=args.add_on_layers_type,
                                mask_theta=args.mask_theta,
                                pretrain_path=args.pretrain_path,
                                hash_code_length=args.hash_code_length)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print("require grad:", name)
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    joint_optimizer_lrs = {'features': args.features_lr,
                        'add_on_layers': args.add_on_layers_lr,
                        'prototype_vectors': args.prototype_vectors_lr,
                        'decoder': args.decoder_lr,}   

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params: {}'.format(n_parameters))

    # timm.optim
    optimizer = create_optimizer(args, model_without_ddp, joint_optimizer_lrs=joint_optimizer_lrs)
    
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"Param Group {i}:")
        print("Parameters:")
        for param in param_group['params']:
            if param.requires_grad:
                print(param.shape)
        print("Config:")
        for key in param_group:
            if key != 'params':  
                print(f"{key}: {param_group[key]}")
        print("\n")
        
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = torch.nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        #model_without_ddp.load_state_dict(checkpoint['model'])
        model_without_ddp.load_state_dict(checkpoint)

        submodules_to_freeze = [model_without_ddp.add_on_layers, model_without_ddp.features, model_without_ddp.decoder, model_without_ddp.cls_tokens_decoder]
        for submodule in submodules_to_freeze:
            for param in submodule.parameters():
                param.requires_grad = False
        # for name, param in model.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")
        
        if args.syn:
            syn_model = construct_PPNet_dino(args, img_size=args.img_size,
                                prototype_shape=args.prototype_shape,
                                num_classes=args.labeled_nums,
                                use_global=args.use_global,
                                global_proto_per_class=args.global_proto_per_class,
                                prototype_activation_function=args.prototype_activation_function,
                                add_on_layers_type=args.add_on_layers_type,
                                mask_theta=args.mask_theta,
                                pretrain_path=args.pretrain_path,
                                hash_code_length=args.hash_code_length)
            syn_model_without_ddp = model
            syn_model_without_ddp.load_state_dict(checkpoint)
            syn_model = syn_model.to(device)
            for param in syn_model.parameters():
                param.requires_grad = False 

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
    
    if args.eval:
        model.eval()
        samples_per_class = args.global_proto_per_class
        class_means = torch.stack([model.prototype_vectors_global[i:i+samples_per_class].mean(0) for i in range(0, model.prototype_vectors_global.size(0), samples_per_class)])
        hash_centers = model.hash_head(class_means)
        test_stats, all_acc_v1, old_acc_v1, new_acc_v1, all_acc_v2, old_acc_v2, new_acc_v2 \
            = evaluate(test_loader_unlabelled, model, args, centers=hash_centers.cpu().sign(), dis_max=get_dis_max(args), epoch=0)  
        #evaluate(test_loader=test_loader_unlabelled, model=model, args=args, centers=hash_centers.cpu().sign(), dis_max=dis_max, epoch=epoch) 
        #logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        # use .2f
        logger.info(f"all_acc_v1: {all_acc_v1 * 100:.2f}, old_acc_v1: {old_acc_v1 * 100:.2f}, new_acc_v1: {new_acc_v1 * 100:.2f}")
        logger.info(f"all_acc_v2: {all_acc_v2 * 100:.2f}, old_acc_v2: {old_acc_v2 * 100:.2f}, new_acc_v2: {new_acc_v2 * 100:.2f}")
        return
    
    if args.eval_train:
        model.eval()
        samples_per_class = args.global_proto_per_class
        class_means = torch.stack([model.prototype_vectors_global[i:i+samples_per_class].mean(0) for i in range(0, model.prototype_vectors_global.size(0), samples_per_class)])
        hash_centers = model.hash_head(class_means)
        test_stats = evaluate_train_dataset(data_loader_train, model, args, centers=hash_centers.cpu().sign(), dis_max=get_dis_max(args), epoch=0)  
        #evaluate(test_loader=test_loader_unlabelled, model=model, args=args, centers=hash_centers.cpu().sign(), dis_max=dis_max, epoch=epoch) 
        #logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        # # use .2f
        # logger.info(f"all_acc_v1: {all_acc_v1 * 100:.2f}, old_acc_v1: {old_acc_v1 * 100:.2f}, new_acc_v1: {new_acc_v1 * 100:.2f}")
        # logger.info(f"all_acc_v2: {all_acc_v2 * 100:.2f}, old_acc_v2: {old_acc_v2 * 100:.2f}, new_acc_v2: {new_acc_v2 * 100:.2f}")
        return

    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):

        train_stats = train_and_evaluate(
                                        model=model, 
                                        data_loader=data_loader_train, 
                                        test_loader_unlabelled=test_loader_unlabelled,
                                        optimizer=optimizer, device=device, 
                                        epoch=epoch, loss_scaler=loss_scaler,
                                        max_norm=args.clip_grad, 
                                        model_ema=model_ema,
                                        args=args,
                                        set_training_mode=True,
                                        syn_model=syn_model if args.syn else None)
        logger.info("Averaged stats:")
        logger.info(train_stats)
        __global_values__["it"] += len(data_loader_train)

        
        lr_scheduler.step(epoch)
        # if args.output_dir:
        #     if (epoch+1) % args.save_ep_freq == 0:
        #         checkpoint_paths = [output_dir / 'checkpoints/checkpoint-{}.pth'.format(epoch)]
        #         for checkpoint_path in checkpoint_paths:
        #             utils.save_on_master({
        #                 'model': model_without_ddp.state_dict(),
        #                 'optimizer': optimizer.state_dict(),
        #                 'lr_scheduler': lr_scheduler.state_dict(),
        #                 'epoch': epoch,
        #                 'model_ema': get_state_dict(model_ema),
        #                 'scaler': loss_scaler.state_dict(),
        #                 'args': args,
        #             }, checkpoint_path)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PHE training', parents=[get_args_parser()])
    args = parser.parse_args()
    # preprocess
    args.prototype_dim = args.zs_dim

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    __global_values__ = dict(it=0)

    valid_super_categories = ['Actinopterygii', 'Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Chromista', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Protozoa', 'Reptilia']

    if args.data_set == 'cub':
        args.data_root = cub_root

    elif args.data_set == 'scars':
        args.data_root = car_root

    elif args.data_set == 'food':
        args.data_root = food_101_root

    elif args.data_set == 'pets':
        args.data_root = oxford_pet_root

    elif args.data_set in valid_super_categories:
        args.data_root = inaturalist_root
 
    args.pretrain_path = pretrain_path

    main(args)