from distutils.log import error
import os
from random import triangular
from tkinter import Y
import ipdb as pdb
import numpy as np
import torch
from scipy.stats import ortho_group, wishart
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

from train_spline import pretrain_spline

root_dir = '.'

def leaky_ReLU_1d(d, negSlope):
    if d > 0:
        return d
    else:
        return d * negSlope

leaky1d = np.vectorize(leaky_ReLU_1d)

def leaky_ReLU(D, negSlope):
    assert negSlope > 0
    return leaky1d(D, negSlope)


def gen_da_data_ortho(
    args,
    Nsegment, 
    Ncomp=4,
    Ncomp_s=2,
    Nlayer=3,
    var_range_l=0.01,
    var_range_r=3,
    mean_range_l=0,
    mean_range_r=3,
    NsegmentObs_train=7500,
    NsegmentObs_test=1000,
    Nobs_test=4096,
    varyMean=True, 
    mixtures=True,
    mixture_from_flow=True,
    flow_training_size=0,
    seed=1,
    n_modes_range_l=2,
    n_modes_range_r=6,
    p_domains_range_l=1,
    p_domains_range_r=2,
    linear_mixing_first=False,
    source = 'Gaussian',
    save_all_datasets=False,
    ):
    """
    generate multivariate data based on the non-stationary non-linear ICA model of Hyvarinen & Morioka (2016)
    we generate mixing matrices using random orthonormal matrices
    INPUT
        - Ncomp: number of components (i.e., dimensionality of the data)
        - Nlayer: number of non-linear layers!
        - Nsegment: number of data segments to generate
        - NsegmentObs: number of observations per segment
        - source: either Laplace or Gaussian, denoting distribution for latent sources
        - NonLin: linearity employed in non-linear mixing. Can be one of "leaky" = leakyReLU or "sigmoid"=sigmoid
          Specifically for leaky activation we also have:
            - negSlope: slope for x < 0 in leaky ReLU
            - Niter4condThresh: number of random matricies to generate to ensure well conditioned
    OUTPUT:
      - output is a dictionary with the following values:
        - sources: original non-stationary source
        - obs: mixed sources
        - labels: segment labels (indicating the non stationarity in the data)
    """

    negSlope = 0.2
    NonLin = 'leaky'
    # np.random.seed(seed)
    randomstate = np.random.RandomState(seed)

    # generate non-stationary data:
    train_size = NsegmentObs_train * Nsegment
    assert Nobs_test == 0 or NsegmentObs_test == 0
    if Nobs_test > 0:
        NsegmentObs_test = int(Nobs_test // Nsegment)
    test_size = NsegmentObs_test * Nsegment
    NsegmentObs_total = NsegmentObs_train + NsegmentObs_test
    Nobs = train_size + test_size  # total number of observations
    labels = np.array([0] * Nobs)  # labels for each observation (populate below)
    Ncomp_c = Ncomp - Ncomp_s

    # generate data, which we will then modulate in a non-stationary manner:
    if source == 'Laplace':
        dat = randomstate.laplace(0, 1, (Nobs, Ncomp))
        dat = scale(dat)  # set to zero mean and unit variance
    elif source == 'Gaussian':
        dat = randomstate.normal(0, 1, (Nobs, Ncomp))
        dat = scale(dat)
    else:
        raise Exception("wrong source distribution")


    # get modulation parameters
    modMat = randomstate.uniform(var_range_l, var_range_r, (Ncomp_s, Nsegment))
    if varyMean:
        meanMat = randomstate.uniform(mean_range_l, mean_range_r, (Ncomp_s, Nsegment))
    else:
        meanMat = np.zeros((Ncomp_s, Nsegment))

    # now we adjust the variance within each segment in a non-stationary manner
    for seg in range(Nsegment):
        segID = range(NsegmentObs_total * seg, NsegmentObs_total * (seg + 1))
        dat[segID, -Ncomp_s:] = np.multiply(dat[segID, -Ncomp_s:], modMat[:, seg])
        dat[segID, -Ncomp_s:] = np.add(dat[segID, -Ncomp_s:], meanMat[:, seg])
        labels[segID] = seg

    # now we are ready to apply the non-linear mixtures:
    mixedDat = np.copy(dat)

    # generate mixing matrices:
    if linear_mixing_first:
        A = ortho_group.rvs(Ncomp, random_state=randomstate)
        mixedDat = np.dot(mixedDat, A)
    for l in range(Nlayer - 1):
        # we first apply non-linear function, then causal matrix!
        if NonLin == 'leaky':
            mixedDat = leaky_ReLU(mixedDat, negSlope)
        elif NonLin == 'sigmoid':
            mixedDat = sigmoidAct(mixedDat)

        # generate causal matrix first:
        A = ortho_group.rvs(Ncomp, random_state=randomstate)  # generateUniformMat( Ncomp, condThresh )
        # apply mixing:
        mixedDat = np.dot(mixedDat, A)

    # stratified split
    x_train, x_test, z_train, z_test, u_train, u_test = train_test_split(
        mixedDat, dat, labels, train_size=train_size, test_size=test_size, random_state=randomstate, stratify=labels,
    )

    if save_all_datasets is True:
        all_datasets = {}
        for domID in range(Nsegment):
            train_indices = u_train<=domID
            test_indices = u_test<=domID
            all_datasets[domID+1] = {
                "train": {"y": z_train[train_indices], "x": x_train[train_indices], "c": u_train[train_indices]},
                "test": {"y": z_test[test_indices], "x": x_test[test_indices], "c": u_test[test_indices]}
            }

        torch.save(all_datasets, f"./data/all_datasets_{Nsegment}_seed_{seed}_domain_validation_size_{NsegmentObs_test}_mean_{mean_range_r}_var_{var_range_r}_n_components_{Ncomp}.pth")
        
        pdb.set_trace()

    return {"y": z_train, "x": x_train, "c": u_train}, {"y": z_test, "x": x_test, "c": u_test}  
