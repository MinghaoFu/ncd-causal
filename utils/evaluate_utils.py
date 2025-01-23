import math
import numpy as np
import os
import seaborn as sns
from scipy.optimize import linear_sum_assignment as linear_assignment


def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def split_cluster_acc_v1(y_true, y_pred, mask):

    """
    Evaluate clustering metrics on two subsets of data, as defined by the mask 'mask'
    (Mask usually corresponding to `Old' and `New' classes in GCD setting)
    :param targets: All ground truth labels
    :param preds: All predictions
    :param mask: Mask defining two subsets
    :return:
    """ 
    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    weight = mask.mean()

    old_acc = cluster_acc(y_true[mask], y_pred[mask])
    new_acc = cluster_acc(y_true[~mask], y_pred[~mask])
    total_acc = weight * old_acc + (1 - weight) * new_acc

    return total_acc, old_acc, new_acc

def split_cluster_acc_v2(y_true, y_pred, mask):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    new_acc /= total_new_instances

    return total_acc, old_acc, new_acc


def binomial_coefficient(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))

def get_dis_max(args):
    Y_U, L = args.unlabeled_nums, args.hash_code_length
    target = (2 ** L) / Y_U

    for d in range(2, L + 1):
        lower_sum = binomial_coefficient(L, d - 2)
        upper_sum = binomial_coefficient(L, d - 1)

        if lower_sum <= target <= upper_sum:
            print("d_max:", d)
            return d

    return 0

# Ours
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.cm import get_cmap
from sklearn.preprocessing import LabelEncoder
from matplotlib.patheffects import withStroke

def visualize_latent_variables(z_list, zs_dim):
    z = torch.cat(z_list, dim=0).detach().cpu().numpy() 
    z_dim = z.shape[-1]
    if zs_dim <= 0 or zs_dim > z_dim:
        print(f"Error: Invalid zs_dim {zs_dim}. It must be in the range [1, {z_dim}].")
        return
    zc_dim = z_dim - zs_dim
    z_mean = np.mean(z, axis=0)
    avg_abs_z = np.mean(np.abs(z - z_mean), axis=0)
    avg_abs_zs = avg_abs_z[:zs_dim]
    avg_abs_zc = avg_abs_z[zs_dim:]
    print(f'Absolute average deviation for zs (dim={zs_dim}): {avg_abs_zs}')
    print(f'Absolute average deviation for zc (dim={zc_dim}): {avg_abs_zc}') 

def visualize_features(features, targets, save_path, seed=42, max_iter=1000, max_classes=10): 
    """
    T-SNE
    Visualizes feature representations with styled clusters on a plain white background,
    showing only the legend and adding a white outline around text.
    Limits the visualization to a maximum of `max_classes` classes.

    Args:
        features (np.ndarray): Feature tensor or array (CLS token features).
        targets (list or np.ndarray): Corresponding labels for the features.
        seed (int): Random seed for reproducibility.
        max_iter (int): Maximum iterations for TSNE.
        max_classes (int): Maximum number of classes to visualize.
    """
    def generate_colors(n):
        """
        Generate `n` distinct colors using a colormap.

        Args:
            n (int): Number of colors to generate.

        Returns:
            list: List of color tuples.
        """
        cmap = get_cmap('tab20')  # You can choose a colormap, e.g., 'tab20', 'viridis', etc.
        return [cmap(i / n) for i in range(n)]
    
    # Reduce the dataset to include only the first `max_classes` unique classes
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(targets)
    unique_classes = np.unique(encoded_labels)
    if len(unique_classes) > max_classes:
        selected_classes = unique_classes[:max_classes]
        mask = np.isin(encoded_labels, selected_classes)
        features = features[mask]
        encoded_labels = encoded_labels[mask]
        targets = np.array(targets)[mask]
    
    n_classes = len(np.unique(encoded_labels))
    colors = generate_colors(n_classes)

    # Apply TSNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=seed, max_iter=max_iter)
    transformed_features = tsne.fit_transform(features)

    text_outline = withStroke(linewidth=3, foreground="white")

    plt.figure(figsize=(10, 8))
    for class_id in range(n_classes):
        indices = (encoded_labels == class_id)
        class_name = label_encoder.inverse_transform([class_id])[0]
        plt.scatter(transformed_features[indices, 0], transformed_features[indices, 1],
                    label=f"{class_name} ({class_id})", s=20, alpha=0.8, color=[colors[class_id]])

        # Place the class number at the center of each cluster with a white outline
        cluster_center = transformed_features[indices].mean(axis=0)
        plt.text(cluster_center[0], cluster_center[1], str(class_id), fontsize=12,
                 ha='center', va='center', color='black', weight='bold',
                 path_effects=[text_outline])

    plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.axis('off')  # Remove axes
    plt.xticks([])   # Remove x-axis ticks
    plt.yticks([])   # Remove y-axis ticks

    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    
    
def draw_class_feature(num_classes, feats, targets, save_path, num_plot_classes, s=30):
    plt.figure(figsize=(16, 6))
    # random select 10 classes from targets
    selected_classes = np.random.choice(num_classes, num_plot_classes, replace=False)
    # get the indices of selected classes
    for cls in selected_classes:
        indices = np.where(np.isin(targets, cls))[0]
        cls_feats = feats[indices]
        feats_mean = np.mean(cls_feats, axis=0) # (num, z_dim)
        
        # plt.plot(z_mean, label=f"Class {cls}",linewidth=0.3)
        sns.distplot(feats_mean, bins=100)
        #plt.scatter(range(len(feats_mean)), feats_mean, label=f"Class {cls}", s=s)  # Use scatter plot with small points
        
    plt.title(f"Mean Values Across Dimensions for Selected Classes {selected_classes}")
    plt.xlabel("Feature Dimension")
    plt.ylabel("Mean Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path + 'mean.png', dpi=1000)
    plt.close()
    
    plt.figure(figsize=(16, 6))
    for cls in selected_classes:
        indices = np.where(np.isin(targets, cls))[0]
        cls_feats = feats[indices]
        feats_std = np.std(cls_feats, axis=0)  # (num, z_dim)
        sns.distplot(feats_std, bins=100)
        #plt.scatter(range(len(feats_var)), feats_var, label=f"Class {cls}", s=s)  # Use scatter plot with small points
        
    plt.title(f"Std of Feature Values Across Dimensions for Selected Classes {selected_classes}")
    plt.xlabel("Feature Dimension")
    plt.ylabel("Std")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path + 'std.png', dpi=1000)
    plt.close()
