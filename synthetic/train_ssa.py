from tokenize import group
import torch
import random
import argparse
import numpy as np
from pathlib import Path
import ipdb as pdb
import os, pwd, yaml
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

from train_spline import pretrain_spline
from sssa import SSA
from utils import load_yaml
from dataset import DANS
from gen_dataset import gen_da_data_ortho
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def main(args):

    assert args.exp is not None, "FATAL: "+__file__+": You must specify an exp config file (e.g., *.yaml)"
    
    current_user = pwd.getpwuid(os.getuid()).pw_name
    script_dir = os.path.dirname(__file__)
    rel_path = os.path.join('./configs', 
                            '%s.yaml'%args.exp)
    abs_file_path = os.path.join(script_dir, rel_path)--
    cfg = load_yaml(abs_file_path)
    print("######### Configuration #########")
    print(yaml.dump(cfg, default_flow_style=False))
    print("#################################")

    # logging
    wandb_logger = WandbLogger(project="partial_identifiability", name=cfg["NAME"])
    #wandb_logger.experiment.config.updte(cfg)

    # generating data
    print("----------We use a freshly generate dataset.-------------")
    train_data, test_data = gen_da_data_ortho(
        args=args,
        Nsegment=cfg["DATA"]["N_DOMAINS"], 
        Ncomp=cfg["DATA"]["N_COMP"],
        Ncomp_s=cfg["DATA"]["N_COMP_S"],
        Nlayer=cfg["DATA"]["N_LAYERS"],
        var_range_l=cfg["DATA"]["VAR_RANGE_L"],
        var_range_r=cfg["DATA"]["VAR_RANGE_R"],
        mean_range_l=cfg["DATA"]["MEAN_RANGE_L"],
        mean_range_r=cfg["DATA"]["MEAN_RANGE_R"],
        NsegmentObs_train=cfg["DATA"]["N_TRAIN_SAMPLES_DOMAIN"],
        NsegmentObs_test=cfg['DATA']['N_TEST_SAMPLES_DOMAIN'],
        Nobs_test=cfg["DATA"]["N_TEST_SAMPLES"],
        varyMean=cfg["DATA"]["VARY_MEAN"], 
        seed=cfg["SEED"],
        mixtures=cfg["DATA"]["MIXTURES"],
        n_modes_range_l=cfg["DATA"]["N_MODES_RANGE_L"],
        n_modes_range_r=cfg["DATA"]["N_MODES_RANGE_R"],
        p_domains_range_l=cfg["DATA"]["P_DOMAINS_RANGE_L"],
        p_domains_range_r=cfg["DATA"]["P_DOMAINS_RANGE_R"],
        mixture_from_flow=cfg["DATA"]["MIXING_FROM_FLOW"],
        flow_training_size=cfg["SPLINE"]["STEPS"]*cfg["SPLINE"]["BS"],
        linear_mixing_first=cfg["DATA"]["LINEAR_MIXING_FIRST"],
        save_all_datasets=cfg["DATA"]["SAVE_ALL_DATASETS"] if "SAVE_ALL_DATASETS" in cfg["DATA"] else False,
    )

    train_dataset, test_dataset = DANS(train_data), DANS(test_data)

    pl.seed_everything(cfg["SEED"])

    # Warm-start spline
    if cfg['SPLINE']['USE_WARM_START']:
        if not os.path.exists(cfg['SPLINE']['PATH']):
            print('Pretraining Spline Flow...', end=' ', flush=True)
            pretrain_spline(args.exp)
            print('Done!')

    train_loader = DataLoader(train_dataset, 
                              batch_size=cfg['VAE']['TRAIN_BS'], 
                              pin_memory=cfg['VAE']['PIN'],
                              num_workers=cfg['VAE']['CPU'],
                              drop_last=False,
                              shuffle=True)

    val_loader = DataLoader(test_dataset, 
                            batch_size=len(test_dataset), 
                            pin_memory=cfg['VAE']['PIN'],
                            num_workers=cfg['VAE']['CPU'],
                            drop_last=True,
                            shuffle=False)

    model = SSA(
        input_dim=cfg["DATA"]["N_COMP"],
        c_dim=cfg["DATA"]["N_COMP"]-cfg["DATA"]["N_COMP_S"],
        s_dim=cfg["DATA"]["N_COMP_S"],
        nclass=cfg["DATA"]["N_DOMAINS"],
        hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
        embedding_dim=cfg['VAE']['EMBEDDING_DIM'],
        bound=cfg['SPLINE']['BOUND'],
        n_flow_layers=cfg['SPLINE']['N_LAYERS'],
        count_bins=cfg['SPLINE']['BINS'],
        order=cfg['SPLINE']['ORDER'],
        beta=cfg['VAE']['BETA'],
        gamma=cfg['VAE']['GAMMA'],
        sigma=cfg['VAE']['SIGMA'],
        vae_slope=cfg['VAE']['SLOPE'],
        lr=cfg['VAE']['LR'],
        use_warm_start=cfg['SPLINE']['USE_WARM_START'],
        spline_pth=cfg['SPLINE']['PATH'],
        decoder_dist=cfg['VAE']['DEC']['DIST'],
        correlation=cfg['MCC']['CORR'],
        encoder_n_layers=cfg['VAE']['ENC']['N_LAYERS'],
        decoder_n_layers=cfg['VAE']['DEC']['N_LAYERS'],
        optimizer=cfg['VAE']['OPTIMIZER'],
        scheduler=cfg['VAE']['SCHEDULER'],
        lr_factor=cfg['VAE']['LR_FACTOR'],
        lr_patience=cfg['VAE']['LR_PATIENCE'],
        hz_to_z=cfg["MCC"]["HZ_TO_Z"] if "HZ_TO_Z" in cfg["MCC"] else False,
    )

    checkpoint_callback = ModelCheckpoint(monitor='val_r2', 
                                          save_top_k=1, 
                                          mode='max')
                  
    trainer = pl.Trainer(
        logger=wandb_logger,
        #gpus=cfg['VAE']['GPU'], 
        accelerator='auto',
        check_val_every_n_epoch=cfg["MCC"]["FREQ"],
        max_epochs=cfg['VAE']['EPOCHS'],
        deterministic=False,
        callbacks=[checkpoint_callback]
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-e',
        '--exp',
        type=str,
        default='test',
    )
    args = argparser.parse_args()
    main(args)
