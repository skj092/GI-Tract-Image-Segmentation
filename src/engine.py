import torch
from tqdm import tqdm
from utils import CFG
from torch import nn



def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice


def train_one_epoch(train_loader, model, loss_fn, optimizer, device):
    model.train()
    running_loss = 0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='training')
    for idx, (xb, yb) in pbar:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logit = model(xb)
        loss = loss_fn(logit, yb)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(train_loss= f'{loss.item():.3f}')
    epoch_loss = running_loss/len(train_loader)
    return epoch_loss

def valid_one_epoch(data_loader, model, loss_fn,device):
    model.eval()
    running_loss = 0
    running_dice = 0
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc='validation')
    for idx, (xb, yb) in pbar:
        xb = xb.to(device)
        yb = yb.to(device)
        logit = model(xb)
        loss = loss_fn(logit, yb)

        # metrix
        logit = nn.Sigmoid()(logit)
        running_dice += dice_coef(yb, logit)

        running_loss += loss.item()
        pbar.set_postfix(valid_loss= f'{loss.item():.3f}')
    epoch_loss = running_loss/len(data_loader)
    epoch_dice = running_dice / len(data_loader)
    return epoch_loss, epoch_dice

