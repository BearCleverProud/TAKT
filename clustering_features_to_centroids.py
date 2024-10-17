import argparse
import os

import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import torch
from scipy.spatial.distance import cdist

from tools.clustering import Kmeans

def multivariate_normal(mean, cov, size=1):
    L = np.linalg.cholesky(cov)
    Z = np.random.normal(size=(cov.shape[0], size))
    X = np.dot(L, Z).T + mean
    return X

def reduce(args, train_list):
    os.makedirs(f'PATH_TO_DATASET_DIR/centroids_{args.dataset}/pt_files/', exist_ok=True)
    os.makedirs(f'PATH_TO_DATASET_DIR/centroids_{args.dataset}/shifts/', exist_ok=True)
    for feat_pth in tqdm(train_list):
        if os.path.isfile(f'PATH_TO_DATASET_DIR/centroids_{args.dataset}/shifts/{feat_pth.split("/")[-1].replace(".pt", "_semantic_shifts.pt")}'):
            continue
        feats = torch.load(feat_pth)
        feats = feats.numpy()

        feats = np.ascontiguousarray(feats, dtype=np.float32)
        kmeans = Kmeans(k=args.num_prototypes, pca_dim=-1)
        kmeans.cluster(feats, seed=66)  # for reproducibility
        assignments = kmeans.labels.astype(np.int64)
        # Check that each cluster has at least two points
        unique_labels, label_counts = np.unique(assignments, return_counts=True)
        if not np.all(label_counts >= 2):
            continue

        # compute the centroids for each cluster
        centroids = np.array([np.mean(feats[assignments == i], axis=0)
                              for i in range(args.num_prototypes)])

        # compute covariance matrix for each cluster
        covariance = np.array([np.cov(feats[assignments == i].T)
                               for i in range(args.num_prototypes)])

        cov_reg = np.zeros_like(covariance)
        for i in range(covariance.shape[0]):
            cov_reg[i] = covariance[i] + 1e-6 * np.eye(covariance.shape[1])
        covariance = cov_reg

        # the semantic shift vectors are enough.
        semantic_shift_vectors = []
        for cov in covariance:
            semantic_shift_vectors.append(
                # sample shift vector from zero-mean multivariate Gaussian distritbuion N(0, cov)
                multivariate_normal(np.zeros(cov.shape[0]), cov,
                                            size=args.num_shift_vectors))

        semantic_shift_vectors = np.array(semantic_shift_vectors)
        torch.save(torch.from_numpy(centroids),
                   f'PATH_TO_DATASET_DIR/centroids_{args.dataset}/pt_files/{feat_pth.split("/")[-1]}')
        torch.save(torch.from_numpy(semantic_shift_vectors),
                   f'PATH_TO_DATASET_DIR/centroids_{args.dataset}/shifts/{feat_pth.split("/")[-1].replace(".pt", "_semantic_shifts.pt")}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='base dictionary construction')
    parser.add_argument('--dataset', type=str, default='nsclc')
    parser.add_argument('--num_prototypes', type=int, default=8)
    parser.add_argument('--num_shift_vectors', type=int, default=200)
    args = parser.parse_args()
    train_root = f'PATH_TO_DATASET_DIR/{args.dataset}_resnet50_256_level_1_without_otsu/pt_files/'
    train_list = os.listdir(train_root)
    train_list = [train_root + each for each in train_list]
    reduce(args, train_list)
