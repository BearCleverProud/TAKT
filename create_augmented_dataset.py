
import os
import torch
from tqdm import tqdm
import numpy as np
from torch import cdist
import argparse
import subprocess
import random

def bash(command):
    subprocess.run(['bash', '-c', command])

def mix_aug(src_feats, tgt_feats, mode='replace', rate=0.3, strength=0.5, shift=None):
    assert mode in ['replace', 'append', 'interpolate', 'cov', 'joint']
    auged_feats = [each for each in src_feats.reshape(-1, 1024)]
    closest_idxs = torch.argmin(cdist(src_feats.reshape(-1, 1024), tgt_feats), axis=1)
    augmented_labels = []
    if mode != 'joint':
        for ix in range(len(src_feats)):
            if np.random.rand() <= rate:
                if mode == 'replace':
                    auged_feats[ix] = tgt_feats[closest_idxs[ix]]
                    augmented_labels.append(ix)
                elif mode == 'append':
                    auged_feats.append(tgt_feats[closest_idxs[ix]])
                    augmented_labels.append(len(auged_feats)-1)
                elif mode == 'interpolate':
                    generated = (1 - strength) * auged_feats[ix] + strength * tgt_feats[closest_idxs[ix]]
                    auged_feats.append(generated)
                    augmented_labels.append(len(auged_feats)-1)
                elif mode == 'cov':
                    generated = auged_feats[ix][np.newaxis, :] + strength * shift[closest_idxs[ix]][np.random.choice(200, 1)]
                    auged_feats.append(generated.flatten())
                    augmented_labels.append(len(auged_feats)-1)
                else:
                    raise NotImplementedError
    else:
        for ix in range(len(src_feats)):
            if np.random.rand() <= rate:
                # replace
                auged_feats[ix] = tgt_feats[closest_idxs[ix]]
                augmented_labels.append(ix)
            if np.random.rand() <= rate:
                # append
                auged_feats.append(tgt_feats[closest_idxs[ix]])
                augmented_labels.append(len(auged_feats)-1)
            if np.random.rand() <= rate:
                # interpolate
                generated = (1 - strength) * auged_feats[ix] + strength * tgt_feats[closest_idxs[ix]]
                auged_feats.append(generated)
                augmented_labels.append(len(auged_feats)-1)
            if np.random.rand() <= rate:
                # covary
                generated = auged_feats[ix][np.newaxis, :] + strength * shift[closest_idxs[ix]][np.random.choice(200, 1)]
                auged_feats.append(generated.flatten())
                augmented_labels.append(len(auged_feats)-1)
    auged_feats = torch.stack(auged_feats, dim=0)
    return auged_feats.cpu(), [i for i in range(len(auged_feats)) if i not in augmented_labels], augmented_labels

if __name__ == '__main__':

    torch.manual_seed(0)

    # Set the random seed for Python's built-in random module
    random.seed(0)

    # Set the random seed for NumPy
    np.random.seed(0)

    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description='base dictionary construction')
    parser.add_argument('--main_dataset', type=str, default='nsclc')
    parser.add_argument('--aux_dataset', type=str, default='camelyon')
    parser.add_argument('--mode', type=str, default='replace')
    parser.add_argument('--rate', type=float, default=0.3)
    parser.add_argument('--strength', type=float, default=0.5)
    args = parser.parse_args()
    
    root_dir = f'PATH/original_datasets/{args.main_dataset}_resnet50_256_level_1_without_otsu/pt_files/'
    new_root = f'PATH/augmented_datasets/{args.main_dataset}_augmented_with_{args.aux_dataset}_{args.mode}_rate_{args.rate}_strength_{args.strength}_with_augmented_labels/pt_files/'
    aug_feat_root = f'PATH/reduced_datasets/reduced_{args.aux_dataset}/pt_files/'
    shift_root = f'PATH/reduced_datasets/reduced_{args.aux_dataset}/shifts/'

    not_aug_dataset = set()
    d = {'nsclc': 'task_2_tcga_nsclc', 'rcc': 'task_1_tcga_rcc', 'camelyon': 'task_3_camelyon'}
    with open(f'splits/{d[args.aux_dataset]}_100/splits_0.csv', 'r') as f:
        f.readline()
        for line in f:
            _, _, val, test = line.split(',')
            if val != '':
                not_aug_dataset.add(val.replace('.svs', '.pt'))
            if test != '':
                not_aug_dataset.add(test.strip().replace('.svs', '.pt'))

    auged_feats = []
    for feature in os.listdir(aug_feat_root):
        if args.aux_dataset == 'camelyon':
            if feature.replace('.pt','') not in not_aug_dataset:
                auged_feats.append(torch.load(f'{aug_feat_root}/{feature}'))
        else:
            if feature not in not_aug_dataset:
                auged_feats.append(torch.load(f'{aug_feat_root}/{feature}'))
    auged_feats = torch.stack(auged_feats, dim=0).cuda()

    shift = []
    for feature in os.listdir(shift_root):
        if args.aux_dataset == 'camelyon':
            if feature.replace('_semantic_shifts.pt','') not in not_aug_dataset:
                shift.append(torch.load(f'{shift_root}/{feature}'))
        else:
            if feature.replace('_semantic_shifts','') not in not_aug_dataset:
                shift.append(torch.load(f'{shift_root}/{feature}'))
    shift = torch.stack(shift, dim=0).cuda()

    auged_feats = auged_feats.reshape(-1, 1024)
    shift = shift.reshape(-1, 200, 1024)
    print(auged_feats.shape, shift.shape)
    os.makedirs(new_root, exist_ok=True)
    features = os.listdir(root_dir)
    features = [root_dir + each for each in features]

    not_aug_dataset = set()
    d = {'nsclc': 'task_2_tcga_nsclc', 'rcc': 'task_1_tcga_rcc', 'camelyon': 'task_3_camelyon'}
    with open(f'splits/{d[args.main_dataset]}_100/splits_0.csv', 'r') as f:
        f.readline()
        for line in f:
            _, _, val, test = line.split(',')
            if val != '':
                not_aug_dataset.add(val.replace('.svs', '.pt'))
            if test != '':
                not_aug_dataset.add(test.strip().replace('.svs', '.pt'))
    
    print(f'{len(not_aug_dataset)} samples in validation and test dataset')
    for feature in tqdm(features):
        if not os.path.isfile(new_root + feature.split('/')[-1]):
            if feature.split('/')[-1] not in not_aug_dataset:
                f = torch.load(feature).cuda()
                new_feats = mix_aug(f, auged_feats, rate=args.rate, strength=args.strength, mode=args.mode, shift=shift)
                torch.save(new_feats, new_root + feature.split('/')[-1])
            else:
                bash(f'cp {root_dir}/{feature.split("/")[-1]} {new_root}/{feature.split("/")[-1]}')