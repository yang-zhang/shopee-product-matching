import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

import gc
from cuml.neighbors import NearestNeighbors

# from sklearn.metrics.pairwise import cosine_similarity
import neighbor

NWKRS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True if DEVICE == torch.device("cuda") else False


def getf1(x, y):
    n = len(np.intersect1d(x, y))
    return 2 * n / (len(x) + len(y))


def getf1s(xs, ys):
    return (getf1(x, y) for x, y in zip(xs, ys))


def meanf1(xs, ys):
    return np.mean(list(getf1s(xs, ys)))


def mk_hsims(df):
    hfeats = df.image_phash.apply(lambda x: bin(int(x, 16))).values
    X = np.array([[eval(o) for o in x[2:]] for x in hfeats], dtype=np.bool)
    n = len(X)
    hdists = np.zeros((n, n), dtype=np.short)
    for i in range(n):
        hdists[:, i] = ((X[:, None, :] != X[i]).sum(2))[:, 0]
    hsims = 1 / (1 + hdists)
    return hsims


def get_preds_by_thrsh(dists, idx, thrsh):
    preds = (ind[dst < thrsh] for dst, ind in zip(dists, idx))
    return preds


def get_preds_from_feats(feats, thrsh):
    dists, idx = neighbor.get_nbrs(feats)
    preds = get_preds_by_thrsh(dists, idx, thrsh)
    return preds


def comb_preds(*preds):
    return (np.unique(np.concatenate(l)) for l in zip(*preds))


def predidx2pids(preds_idx, pids):
    return [pids[o] for o in preds_idx]


def get_targets(df):
    grp2ids = df.groupby("label_group").posting_id.agg("unique").to_dict()
    targets = df.label_group.map(grp2ids)
    return targets


def preds2pids(preds, pids):
    return [pids[o] for o in preds]


def get_preds_pids_h(df):
    hsh2ids = df.groupby("image_phash").posting_id.agg("unique").to_dict()
    preds_h = df.image_phash.map(hsh2ids)
    return preds_h


# http://localhost:8080/notebooks/git/shopee-product-matching/working/dev0021--rapidai-nearest-neighbor-combine.ipynb
thrsh_v = 6.85
thrsh_t = 6.1
