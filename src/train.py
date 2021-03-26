import random
import numpy as np
import torch
from tqdm import tqdm


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, optimizer, criterion):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target) in bar:

        data, target = data, target
        optimizer.zero_grad()

        logits_m = model(data)
        loss = criterion(logits_m, target)
        loss.backward()
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description("loss: %.5f, smth: %.5f" % (loss_np, smooth_loss))

    return train_loss


def val_epoch(model, valid_loader, criterion):

    model.eval()
    val_loss = []
    PRODS_M = []
    PREDS_M = []
    TARGETS = []

    with torch.no_grad():
        for (data, target) in tqdm(valid_loader):
            data, target = data, target

            logits_m = model(data)

            lmax_m = logits_m.max(1)
            probs_m = lmax_m.values
            preds_m = lmax_m.indices

            PRODS_M.append(probs_m.detach().cpu())
            PREDS_M.append(preds_m.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(logits_m, target)
            val_loss.append(loss.detach().cpu().numpy())

        val_loss = np.mean(val_loss)
        PRODS_M = torch.cat(PRODS_M).numpy()
        PREDS_M = torch.cat(PREDS_M).numpy()
        TARGETS = torch.cat(TARGETS)

        acc_m = (PREDS_M == TARGETS.numpy()).mean() * 100.0
        y_true = {
            idx: target if target >= 0 else None for idx, target in enumerate(TARGETS)
        }
        y_pred_m = {
            idx: (pred_cls, conf)
            for idx, (pred_cls, conf) in enumerate(zip(PREDS_M, PRODS_M))
        }
        return val_loss, acc_m
