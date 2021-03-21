import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

import gc
from sklearn.metrics.pairwise import cosine_similarity

NWKRS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True if torch.cuda.is_available() else False


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


def get_preds_by_thrsh(sims, thrsh):
    isclose = sims >= thrsh
    preds = (np.where(x)[0] for x in isclose)
    return preds


def preds2pids(preds, pids):
    return [pids[o] for o in preds]


# git/shopee-product-matching/working/dev0015--threshold--combine.ipynb
thrsh_v = 0.81
thrsh_t = 0.95
thrsh_h = 0.11
