import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import os
import wandb 
import copy
from scipy.special import comb
from timm.utils import NativeScaler, ModelEma

import torch.nn.functional as F

from tqdm import tqdm

from utils import split_cluster_acc_v1, split_cluster_acc_v2, get_dis_max, SmoothedValue, MetricLogger, \
    visualize_latent_variables, visualize_features, get_mask, supervised_contrastive_loss, draw_class_feature

############################################################### Ours ######################################################################
def ortho_loss(z):
    """Compute penalty to enforce independence via covariance"""
    batch_size, dim = z.size()

    z_centered = z - z.mean(dim=0)
    cov_matrix = torch.mm(z_centered.t(), z_centered) / (batch_size - 1)
    off_diag = cov_matrix - torch.diag(torch.diag(cov_matrix))
    penalty = torch.norm(off_diag, p='fro')
    return penalty.sum() / batch_size

def recon_loss(x, x_recon, distribution='gaussian'):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(
            x_recon, x, reduction='sum').div(batch_size)

    elif distribution == 'gaussian':
        recon_loss = F.mse_loss(x_recon, x, reduction='sum').div(batch_size)

    elif distribution == 'sigmoid_gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum').div(batch_size)

    return recon_loss

def intra_class_zc(zc, n_intra, temperature=1.0):
    # logits shape: (n_intra * batch_size, num_classes) 
    batch_size, zc_dim = zc.size()
    zc = zc.view(n_intra, -1, zc_dim) / temperature
    intra_loss = 0
    for view_i in range(n_intra):
        zc_view_i = zc[view_i]
        for view_j in range(view_i + 1, n_intra):
            zc_view_j = zc[view_j]
            # reduction='batchmean': divide the calculated result by num_heads=4,
            # then the result will be divided by the number of labelled samples and labelled classes
            intra_mse_loss = F.mse_loss(zc_view_i, zc_view_j, reduction='sum') / (batch_size * zc_dim)    
            # then the result will be divided by the number of unlabelled samples and unlabelled classes
            intra_loss += intra_mse_loss
    return intra_loss / comb(n_intra, 2, exact=True)

def intra_class_sKLD(logits, n_intra, temperature=1.0):
    # logits shape: (n_intra * batch_size, num_classes) 
    batch_size, num_classes = logits.size()
    logits = logits.view(n_intra, -1, num_classes)   
    log_prob_lab = F.log_softmax(logits / temperature, dim=-1)

    kl_loss = 0
    for view_i in range(n_intra):
        log_prob_lab_view_i = log_prob_lab[view_i]
        for view_j in range(view_i + 1, n_intra):
            log_prob_lab_view_j = log_prob_lab[view_j]
            # reduction='batchmean': divide the calculated result by num_heads=4,
            # then the result will be divided by the number of labelled samples and labelled classes
            kl_loss_lab = (F.kl_div(log_prob_lab_view_i, torch.exp(log_prob_lab_view_j), reduction='batchmean')
                            + F.kl_div(log_prob_lab_view_j, torch.exp(log_prob_lab_view_i), reduction='batchmean')
                            ) / (batch_size * num_classes)
            # then the result will be divided by the number of unlabelled samples and unlabelled classes
            kl_loss += kl_loss_lab
    return kl_loss / comb(n_intra, 2, exact=True)

def compute_independence_loss(features, mask=None, p=0.5):
    if mask is not None:
        features = features * mask
    batch_size, dim = features.size()   
    features = (features - features.mean(dim=0)) / features.std(dim=0)
    covariance_matrix = torch.mm(features.T, features) / (features.size(0) - 1)
    diag = torch.diag(covariance_matrix)
    stddev_matrix = torch.sqrt(diag.unsqueeze(1) * diag)
    correlation_matrix = covariance_matrix / stddev_matrix
    diag_mask = 1 - torch.eye(correlation_matrix.size(0), device=features.device)
    off_diag_loss = (correlation_matrix * diag_mask).pow(2).sum() / batch_size
    return off_diag_loss

##############################################################################################################################################

def cos_eps_loss(u, y, hash_center):

    u_norm = F.normalize(u)
    centers_norm = F.normalize(hash_center)
    cos_sim = torch.matmul(u_norm, torch.transpose(centers_norm, 0, 1)) # batch x n_class

    loss = torch.nn.CrossEntropyLoss()(cos_sim, y)

    return loss

def sep_loss(protop_centers, L = 12, dis_max = 3, alpha=0.95):
    labels = torch.arange(protop_centers.shape[0])
    dot_product = torch.matmul(protop_centers, protop_centers.T)
    hamming_distance = 0.5 * (L*alpha - dot_product)
    mask_diff = labels.unsqueeze(1) != labels.unsqueeze(0)
    mask_diff = mask_diff.cuda()
    loss_sep = (F.relu(dis_max - hamming_distance) * mask_diff.float()).sum(-1)
    return loss_sep.mean()

def test_time_training(model, images, label, num_steps=100):
    base_model = copy.deepcopy(model)
    
    base_model.eval()
    
    # Example: Enable gradients only for 'layer3' and 'layer4'
    # Define a list of layer names for which you want to enable gradients
    layers_to_train = ['add_on_layers', 'cls_tokens_decoder']

    # Iterate over all named parameters
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_train):  # Check if the layer name matches
            param.requires_grad = True  # Enable gradients
        else:
            param.requires_grad = False  # Disable gradients

    model.train()
    
    images = images.cuda() 
    label = label.cuda() 
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    loss_scaler = NativeScaler()
    max_norm = 0.1

    for i in tqdm(range(num_steps)):
        optimizer.zero_grad()
        logits, vit, hash_feat, frz_cls_tokens, frz_patch_tokens, cls_tokens, patch_tokens, zc = model(images) # cls_tokens is z
        loss = recon_loss(frz_cls_tokens, cls_tokens) 
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters())
        print(f"loss: {loss.item()}")   
    
    model.eval()
    return model


def train_and_evaluate(model, 
                    data_loader, 
                    test_loader_unlabelled, 
                    optimizer, device, 
                    epoch, loss_scaler, 
                    max_norm, 
                    model_ema, 
                    args=None, 
                    set_training_mode=True,
                    syn_model=None):

    wandb.login(key='792a3b819c6b832d0087bd4905542a95c7236076')  # Add wandb login
    wandb.init(name=args.output_dir)
    
    model.train(set_training_mode)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 30

    logger = logging.getLogger("train")
    logger.info("Start train one epoch")
    it = 0

    dis_max = get_dis_max(args)

    for batch_index, (samples, targets, _, ind) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if args.n_intra > 1:
            samples = torch.cat(samples, dim=0)
            targets = targets.repeat(args.n_intra)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        batch_size = samples.shape[0]

        with torch.cuda.amp.autocast():
            if args.resume and args.syn:
                syn_model.eval()
                hash_feat, frz_cls_tokens, frz_patch_tokens, cls_tokens, patch_tokens, zc = syn_model(samples)
                #f_cls = syn_model.cls_tokens_decoder(torch.cat([cls_tokens + torch.randn_like(cls_tokens), zc], dim=1))
                
                # topk_values, topk_indices = cls_tokens.topk(20, dim=-1)
                # mask = torch.ones_like(cls_tokens, dtype=torch.bool)
                # mask.scatter_(1, topk_indices, False)
                # cls_tokens = cls_tokens * mask.float()
                # f_cls = syn_model.cls_tokens_decoder(torch.cat([cls_tokens, zc], dim=1))
                
                f_cls = exchange_topk_features(cls_tokens, targets, args.unlabeled_nums, k_top=200)   
                
                syn_frz_cls_tokens = syn_model.decoder(torch.cat([f_cls.unsqueeze(1), patch_tokens], dim=1)).reshape(f_cls.shape)
            
            if args.resume and args.syn and (epoch + 1) // 3 == 0:# and (epoch + 1) // 4 != 0:
                logits, vit, hash_feat, frz_cls_tokens, frz_patch_tokens, cls_tokens, patch_tokens, zc = model(samples, syn_frz_cls_tokens)
                # targets repeat * 2
            else:
                logits, vit, hash_feat, frz_cls_tokens, frz_patch_tokens, cls_tokens, patch_tokens, zc = model(samples)

            # reconstruction
            f_cls = model.cls_tokens_decoder(torch.cat([cls_tokens, zc], dim=1)) # cls tokens is zs 
            cls_tokens_hat, patch_tokens_hat = model.decoder(torch.cat([f_cls.unsqueeze(1), patch_tokens.detach()], dim=1))
            cls_recon_loss = recon_loss(frz_cls_tokens, cls_tokens_hat)
            patch_recon_loss = recon_loss(frz_patch_tokens, patch_tokens_hat)  
            
            cls_distance = 1 - F.cosine_similarity(frz_cls_tokens, cls_tokens_hat, dim=-1) # compute cos distance for evaluate quality 
            
            # recon z from hash
            mean = model.flow(hash_feat)
            std = torch.exp(0.5 * model.logvar)
            #eps = torch.rand_like(std)
            eps = torch.zeros_like(std)
            flow_z = mean + eps * std
            z_recon_loss = recon_loss(cls_tokens, flow_z)

            # independence 
            eps_list = []
            
            for i in range(args.hash_code_length):
                eps_list.append(model.hash_flows[i](torch.cat([hash_feat[:, i:i+1], targets.unsqueeze(1)], dim=-1)))
            eps = torch.cat(eps_list, dim=1) 
            hash_mask = get_mask(model.eps_d_mask_logits)       
            hash_ind_loss = compute_independence_loss(eps, hash_mask)  

            # Sparsity loss on latent variable z
            sparsity_loss = torch.mean(torch.abs(cls_tokens))

            # independence
            zc_ind_loss = compute_independence_loss(zc)
            
            # supervise loss
            probs = F.log_softmax(logits, dim=1)
            loss_protop = torch.nn.NLLLoss()(probs, targets)

            # train acc
            train_accuracy = (torch.argmax(logits, dim=1) == targets).float().mean().item()

            ## get hash centers
            samples_per_class = args.global_proto_per_class
            class_means = torch.stack([model.prototype_vectors_global[i:i+samples_per_class].mean(0) for i in range(0, model.prototype_vectors_global.size(0), samples_per_class)])
            hash_centers = model.hash_head(class_means)

            hash_centers_sign = torch.nn.Tanh()(hash_centers*3)

            # hash centers separation loss
            loss_sep = sep_loss(hash_centers_sign, L=args.hash_code_length, dis_max=dis_max)

            # hash center quantization loss
            loss_quan = (1 - torch.abs(hash_centers_sign)).mean() 

            ## hash feature optimize loss
            loss_feature = cos_eps_loss(hash_feat, targets, hash_centers)

            # recon hash from y
            gen_hash, y_embed = model.gen_hash(targets.unsqueeze(1), torch.randn(batch_size, args.hash_code_length).to(device) * 0.1)   
            #hash_recon_loss = recon_loss(hash_centers_sign, gen_hash)    
            hash_recon_loss = recon_loss(hash_feat, gen_hash)    
            y_contrastive_loss = supervised_contrastive_loss(args)(y_embed, targets)    
            
            loss = (
                loss_protop
                + loss_sep * args.alpha
                + loss_quan * args.alpha
                + loss_feature * args.beta
                #+ zc_ind_loss * args.l_ind
                + hash_ind_loss * args.l_ind
                + z_recon_loss * 1e-4
                + hash_recon_loss * 1e-4
                + y_contrastive_loss
                + sparsity_loss * args.l_spa
            )
            if args.syn is False:
                loss = loss + cls_recon_loss * args.l_recon / 2 + patch_recon_loss * args.l_recon / 197 * 2
            wandb.log({
                'loss_protop': loss_protop.item(),
                'loss_feature': loss_feature.item(),
                'loss_sep': loss_sep.item(),
                'loss_quan': loss_quan.item(),
                'cls_recon_loss': cls_recon_loss.item(),    
                'sparsity_loss': sparsity_loss.item(),
                'zc_ind_loss': zc_ind_loss.item(),
                'hash_recon_loss': hash_recon_loss.item(),  
                'z_recon_loss': z_recon_loss.item(),  
                'train_accuracy': train_accuracy,
                'cls_distance': cls_distance.mean().item(),
                'y_contrastive_loss': y_contrastive_loss.item(),    
                'patch_recon_loss': patch_recon_loss.item(),    
                'lr': optimizer.param_groups[0]["lr"],
            })
            if args.n_intra > 1:
                #infra_class_loss = intra_class_sKLD(zc, args.n_intra, temperature=1.0) * args.l_intra
                infra_class_loss = intra_class_zc(zc, args.n_intra, temperature=1.0) * args.l_intra
                loss += infra_class_loss   
                
                wandb.log({'intra_loss': infra_class_loss.item()})  
                metric_logger.update(intra_loss=infra_class_loss.item())    
            
        optimizer.zero_grad()

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        # metric_logger.update(loss=loss_value)
        metric_logger.update(loss_protop=loss_protop.item())
        metric_logger.update(loss_feature=loss_feature.item())
        metric_logger.update(loss_sep=loss_sep.item())        
        metric_logger.update(loss_quan=loss_quan.item())
        metric_logger.update(sparsity_loss=sparsity_loss.item())
        metric_logger.update(zc_ind_loss=zc_ind_loss.item())
        metric_logger.update(z_recon_loss=z_recon_loss.item())
        metric_logger.update(cls_recon_loss=cls_recon_loss.item())
        metric_logger.update(cls_distance=cls_distance.mean().item())
        metric_logger.update(train_accuracy=train_accuracy)

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


        it += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    _, all_acc_v1, old_acc_v1, new_acc_v1, all_acc_v2, old_acc_v2, new_acc_v2 = \
        evaluate(test_loader=test_loader_unlabelled, model=model, args=args, centers=hash_centers.cpu().sign(), dis_max=dis_max, epoch=epoch, TTT=args.ttt) 
    wandb.log({      
        'all_acc_v1': all_acc_v1,
        'all_acc_v2': all_acc_v2,
    })
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compute_hamming_distance_list(list1, list2):
    """Compute the Hamming distance between two lists"""
    # Use list comprehension and zip to compare if corresponding elements are different
    differences = [x != y for x, y in zip(list1, list2)]
    # Count the number of differing elements, which is the Hamming distance
    hamming_distance = sum(differences)
    return hamming_distance

def syn_data():
    pass

best_v1 = 0 
best_v2 = 0
#@torch.no_grad()
def evaluate(test_loader, model, args, centers, dis_max, epoch, TTT=False):
    global best_v1, best_v2

    radius = max(math.floor(dis_max / 2), 1)
    logger = logging.getLogger("validate")
    logger.info("Start validation")
    logger.info(f"Radius: {radius}")

    metric_logger = MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])
    z_list = []
    zs_list = []
    zc_list = []
    frz_cls_h_list = []
    frz_cls_hat_list = []
    label_lst = []
    test_loss_quan = []

    for batch_idx, (images, label, _, _) in enumerate(tqdm(test_loader)):
        images = images.cuda() 
        label = label.cuda() 
        
        if TTT:
            base_model = copy.deepcopy(model)   
            model = test_time_training(model, images, label, num_steps=1)
        with torch.no_grad():
            feats, frz_cls_tokens, frz_patch_tokens, cls_tokens, patch_tokens, zc = model(images) # cls_tokens is z

            hash_sign = torch.nn.Tanh()(feats*3)
            test_loss_quan = (1 - torch.abs(hash_sign)).mean()
            #print('test_loss_quan:', test_loss_quan)
            frz_cls_hat = model.cls_tokens_decoder(torch.cat([cls_tokens, zc], dim=1))
            frz_cls_h_list.append(F.normalize(frz_cls_tokens).detach().cpu().numpy())
            frz_cls_hat_list.append(F.normalize(frz_cls_hat).detach().cpu().numpy())
            z_list.append(F.normalize(torch.cat([cls_tokens, zc], dim=1)).detach().cpu().numpy()) 
            zs_list.append(F.normalize(cls_tokens).detach().cpu().numpy())  
            zc_list.append(F.normalize(zc).detach().cpu().numpy())
            all_feats.append(feats.cpu().numpy())
            label_lst.append(label.cpu().numpy())   

            targets = np.append(targets, label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(args.labeled_nums) else False for x in label]))
            
        if TTT:
            model = base_model
    with torch.no_grad():
        all_feats = np.concatenate(all_feats, axis=0)
        all_frz_cls_h = np.concatenate(frz_cls_h_list, axis=0) 
        all_frz_cls_hat = np.concatenate(frz_cls_hat_list, axis=0) 
        all_z = np.concatenate(z_list, axis=0)  
        all_zs = np.concatenate(zs_list, axis=0)
        all_zc = np.concatenate(zc_list, axis=0)
        all_label = np.concatenate(label_lst, axis=0)
        if args.eval:
            if args.zc_dim > 0: 
                visualize_features(all_zc, targets, os.path.join(args.output_dir, "z_c.png"), seed=42, max_iter=1000)
            # visualize latent variables
            draw_class_feature(args.unlabeled_nums, all_feats, targets, os.path.join(args.output_dir, "class_hash"), num_plot_classes=3) 
            # save all zs to numpy 
            np.save(os.path.join(args.output_dir, "zs.npy"), all_zs)
            np.save(os.path.join(args.output_dir, "label.npy"), all_label)      
            # visualize z
            visualize_features(all_frz_cls_h, targets, os.path.join(args.output_dir, "cls_tokens.png"), seed=42, max_iter=1000)    
            visualize_features(all_frz_cls_hat, targets, os.path.join(args.output_dir, "cls_tokens_hat.png"), seed=42, max_iter=1000)   
            visualize_features(all_z, targets, os.path.join(args.output_dir, "z.png"), seed=42, max_iter=1000)  
            visualize_features(all_zs, targets, os.path.join(args.output_dir, "zs.png"), seed=42, max_iter=1000)
        
        # Hash
        feats_hash = torch.Tensor(all_feats > 0).float().tolist()

        hash_dict = centers.numpy().tolist()

        # Store the category index corresponding to each feature
        preds1 = []  

        for feat in feats_hash:
            found = False
            # First check if the same category index already exists
            if feat in hash_dict:
                # Use the index of that category
                preds1.append(hash_dict.index(feat))  
                found = True
                
            if not found:
                # If no identical category index is found, then judge based on distance
                distances = [compute_hamming_distance_list(feat, center) for center in hash_dict]
                min_distance = min(distances)
                min_index = distances.index(min_distance)

                if min_distance <= radius:
                    preds1.append(min_index)
                    found = True

            if not found:
                # If the distance between feat and all existing categories exceeds the Hamming sphere radius, create a new category
                hash_dict.append(feat) 
                # Use the index of the new category as the classification result
                preds1.append(len(hash_dict) - 1)

        preds1 = np.array(preds1)

        all_acc_v1, old_acc_v1, new_acc_v1 = split_cluster_acc_v1(y_true=targets, y_pred=preds1, mask=mask)
        logger.info(f'test len(list(set(preds1))): {len(list(set(preds1)))} len(preds): {len(preds1)}')
        logger.info(f"Evaluate V1: all_acc: {all_acc_v1 * 100:.2f} old_acc: {old_acc_v1 * 100:.2f} new_acc: {new_acc_v1 * 100:.2f}")

        all_acc_v2, old_acc_v2, new_acc_v2 = split_cluster_acc_v2(y_true=targets, y_pred=preds1, mask=mask)
        logger.info(f"Evaluate V2: all_acc: {all_acc_v2 * 100:.2f} old_acc: {old_acc_v2 * 100:.2f} new_acc: {new_acc_v2 * 100:.2f}")
        
        if best_v1 < all_acc_v1:    
            best_v1 = all_acc_v1
        if best_v2 < all_acc_v2:
            best_v2 = all_acc_v2
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'checkpoints', f"best_model.pth"))
            with open(os.path.join(args.output_dir, 'checkpoints', f"description.txt"), 'w') as file:
                file.write("all_acc_v1: " + str(all_acc_v1) + "\n" + "old_acc_v1: " + str(old_acc_v1) + "\n" + "new_acc_v1: " + str(new_acc_v1) + "\n" + "all_acc_v2: " + str(all_acc_v2) + "\n" + "old_acc_v2: " + str(old_acc_v2) + "\n" + "new_acc_v2: " + str(new_acc_v2) + "\n" + "epoch: " + str(epoch) + "\n")       
        
        logger.info(f"Best Evaluate V1: all_acc: {best_v1 * 100:.2f}, Best Evaluate V2: all_acc: {best_v2 * 100:.2f}")    
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, all_acc_v1, old_acc_v1, new_acc_v1, all_acc_v2, old_acc_v2, new_acc_v2

def evaluate_train_dataset(train_loader, model, args, centers, dis_max, epoch):
    global best_v1, best_v2

    radius = max(math.floor(dis_max / 2), 1)
    logger = logging.getLogger("validate")
    logger.info("Start validation")
    logger.info(f"Radius: {radius}")

    metric_logger = MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])
    z_list = []
    zs_list = []
    zc_list = []
    frz_cls_h_list = []
    frz_cls_hat_list = []
    label_lst = []
    test_loss_quan = []

    for batch_idx, (images, label, _, _) in enumerate(tqdm(train_loader)):
        images = images.cuda() 
        label = label.cuda() 
        feats, frz_cls_tokens, frz_patch_tokens, cls_tokens, patch_tokens, zc = model(images) # cls_tokens is z
        
        
        hash_sign = torch.nn.Tanh()(feats*3)
        test_loss_quan = (1 - torch.abs(hash_sign)).mean()
        #print('test_loss_quan:', test_loss_quan)

        frz_cls_hat = model.cls_tokens_decoder(torch.cat([cls_tokens, zc], dim=1))
        frz_cls_h_list.append(F.normalize(frz_cls_tokens).detach().cpu().numpy())
        frz_cls_hat_list.append(F.normalize(frz_cls_hat).detach().cpu().numpy())
        z_list.append(F.normalize(torch.cat([cls_tokens, zc], dim=1)).detach().cpu().numpy()) 
        zs_list.append(F.normalize(cls_tokens).detach().cpu().numpy())  
        zc_list.append(F.normalize(zc).detach().cpu().numpy())
        all_feats.append(feats.detach().cpu().numpy())
        label_lst.append(label.cpu().numpy())   

        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(args.labeled_nums) else False for x in label]))
    
    all_feats = np.concatenate(all_feats, axis=0)
    all_frz_cls_h = np.concatenate(frz_cls_h_list, axis=0) 
    all_frz_cls_hat = np.concatenate(frz_cls_hat_list, axis=0) 
    all_z = np.concatenate(z_list, axis=0)  
    all_zs = np.concatenate(zs_list, axis=0)
    all_zc = np.concatenate(zc_list, axis=0)
    all_label = np.concatenate(label_lst, axis=0)   
    if args.eval_train:
        if args.zc_dim > 0: 
            visualize_features(all_zc, targets, os.path.join(args.output_dir, "z_c.png"), seed=42, max_iter=1000)
        # visualize latent variables
        draw_class_feature(args.unlabeled_nums, all_feats, targets, os.path.join(args.output_dir, "class_hash"), num_plot_classes=3) 
        # save all zs to numpy 
        np.save(os.path.join(args.output_dir, "train_zs.npy"), all_zs)
        np.save(os.path.join(args.output_dir, "train_label.npy"), all_label)    
        # visualize z
        visualize_features(all_frz_cls_h, targets, os.path.join(args.output_dir, "cls_tokens.png"), seed=42, max_iter=1000)    
        visualize_features(all_frz_cls_hat, targets, os.path.join(args.output_dir, "cls_tokens_hat.png"), seed=42, max_iter=1000)   
        visualize_features(all_z, targets, os.path.join(args.output_dir, "z.png"), seed=42, max_iter=1000)  
        visualize_features(all_zs, targets, os.path.join(args.output_dir, "zs.png"), seed=42, max_iter=1000)
    
    # Hash
    feats_hash = torch.Tensor(all_feats > 0).float().tolist()

    hash_dict = centers.detach().cpu().numpy().tolist()

    # Store the category index corresponding to each feature
    preds1 = []  

    for feat in feats_hash:
        found = False
        # First check if the same category index already exists
        if feat in hash_dict:
            # Use the index of that category
            preds1.append(hash_dict.index(feat))  
            found = True
            
        if not found:
            # If no identical category index is found, then judge based on distance
            distances = [compute_hamming_distance_list(feat, center) for center in hash_dict]
            min_distance = min(distances)
            min_index = distances.index(min_distance)

            if min_distance <= radius:
                preds1.append(min_index)
                found = True

        if not found:
            # If the distance between feat and all existing categories exceeds the Hamming sphere radius, create a new category
            hash_dict.append(feat) 
            # Use the index of the new category as the classification result
            preds1.append(len(hash_dict) - 1)

    preds1 = np.array(preds1)
    
    # all_acc_v1, old_acc_v1, new_acc_v1 = split_cluster_acc_v1(y_true=targets, y_pred=preds1, mask=mask)
    # logger.info(f'test len(list(set(preds1))): {len(list(set(preds1)))} len(preds): {len(preds1)}')
    # logger.info(f"Evaluate V1: all_acc: {all_acc_v1 * 100:.2f} old_acc: {old_acc_v1 * 100:.2f} new_acc: {new_acc_v1 * 100:.2f}")

    # all_acc_v2, old_acc_v2, new_acc_v2 = split_cluster_acc_v2(y_true=targets, y_pred=preds1, mask=mask)
    # logger.info(f"Evaluate V2: all_acc: {all_acc_v2 * 100:.2f} old_acc: {old_acc_v2 * 100:.2f} new_acc: {new_acc_v2 * 100:.2f}")
    
    # if best_v1 < all_acc_v1:    
    #     best_v1 = all_acc_v1
    # if best_v2 < all_acc_v2:
    #     best_v2 = all_acc_v2
    #     torch.save(model.state_dict(), os.path.join(args.output_dir, 'checkpoints', f"best_model.pth"))
    #     with open(os.path.join(args.output_dir, 'checkpoints', f"description.txt"), 'w') as file:
    # #         file.write("all_acc_v1: " + str(all_acc_v1) + "\n" + "old_acc_v1: " + str(old_acc_v1) + "\n" + "new_acc_v1: " + str(new_acc_v1) + "\n" + "all_acc_v2: " + str(all_acc_v2) + "\n" + "old_acc_v2: " + str(old_acc_v2) + "\n" + "new_acc_v2: " + str(new_acc_v2) + "\n" + "epoch: " + str(epoch) + "\n")       
    
    # logger.info(f"Best Evaluate V1: all_acc: {best_v1 * 100:.2f}, Best Evaluate V2: all_acc: {best_v2 * 100:.2f}")    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# synthesize feature with class-wise topk val replacement
def exchange_topk_features(ori_features, labels, num_classes, k_top=200):
    mu_f = torch.stack([ori_features[labels == c].mean(0) for c in range(num_classes)])
    std_f = torch.stack([ori_features[labels == c].std(0) for c in range(num_classes)])
    
    topk_ind = torch.stack([mu.topk(k=k_top).indices for mu in mu_f], dim=0)
    topk_val = torch.stack([mu.topk(k=k_top).values for mu in mu_f], dim=0)

    # Mixing method and swap method
    method = 'category'  # Options: 'instance', 'category'
    swap_method = 'a'  # Options: 'a', 'b'
    mixed_features = torch.zeros(
        ori_features.size(0), num_classes, ori_features.size(1), 
        device=ori_features.device
    )  # Shape: samples x classes x dimensions
    for i in tqdm(range(ori_features.size(0))):
        import pdb; pdb.set_trace() 
        line = ori_features[i].clone()
        class_a = labels[i].item()
        for class_b in range(num_classes):
            if method == 'instance':
                class_mask = torch.where(labels == class_b)[0]
                sample_b = class_mask[torch.randint(len(class_mask), (1,))].item()
                feat_b = ori_features[sample_b]
            elif method == 'category':
                feat_b = mu_f[class_b]
            else:
                raise NotImplementedError("Unsupported method type")

            replace_ind = topk_ind[class_a if swap_method == 'a' else class_b]
            line[replace_ind] = feat_b[replace_ind]
            mixed_features[i, class_b] = line

    return mixed_features