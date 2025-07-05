# Import necessary libraries
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import segmentation_models_pytorch as smp
from torch import optim
from dataset import prepare_loaders
from utils import data_transforms, CFG, set_seed, plot_batch
from engine import train_one_epoch, valid_one_epoch
import wandb
import time
from torch import nn
import torch



if __name__ == "__main__":
    device = CFG.device
    # set seed for reproducibility
    set_seed(CFG.seed)
    fold = 0

    # log experiment using wandb
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        api_key = user_secrets.get_secret("wandb")
        wandb.login(key=api_key)
        anonymous = None
    except:
        anonymous = "must"
        print('using anonymous')

    # Data Preprocessing
    path = Path('data/preprocessed')
    df = pd.read_csv(path/'train.csv')
    df['segmentation'] = df['segmentation'].fillna('')
    df['mask_path'] = df['mask_path'].str.replace('/kaggle/input/uwmgi-mask-dataset/png/', str(path/'np')).str.replace('png', 'npy')
    df['image_path'] = df['image_path'].str.replace('/kaggle/input/uw-madison-gi-tract-image-segmentation', str(path.parent))
    df['rle_len'] = df['segmentation'].apply(len)
    df['empty'] = (df.rle_len == 0)
    # print(df.head())


    # K-Fold Cross Validation
    skf = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(df, df['empty'], groups=df['case'])):
        df.loc[valid_idx, 'fold'] = fold

    # print(df.groupby(['fold', 'empty'])['id'].count())

    train_loader, valid_loader = prepare_loaders(df, fold, data_transforms, CFG, debug=True)
    # imgs, msks = next(iter(train_loader))
    # print("shape of one batch", imgs.size(), msks.size())
    # plot_batch(imgs, msks, 5)

    model = smp.Unet(encoder_name=CFG.backbone, encoder_weight='imagenet', classes=CFG.n_class, activation=None)
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr)
    loss_fn = smp.losses.SoftBCEWithLogitsLoss()

    # y_pred = model(imgs) # (bs, 3, 224, 224)
    # y_pred = nn.Sigmoid()(y_pred)

    tik = time.time()
    for epoch in range(3):
        train_loss = train_one_epoch(train_loader, model, loss_fn, optimizer, device)
        valid_loss, valid_dice = valid_one_epoch(valid_loader, model, loss_fn, device)
        print(f"Epoch --- {epoch}")
        print(f"train_loss = {train_loss:.3f}, valid_loss = {valid_loss:.3f}, valid_dice = {valid_dice:.3f}")
    tok = time.time()
    print(f"Total time taken: {tok-tik:.2}s")

    # save model
    PATH = f"last_epoch-{fold:02d}.bin"
    torch.save(model.state_dict(), PATH)

    # prediction
    xb, yb = next(iter(valid_loader))
    xb = xb.to(device)
    yb = yb.to(device)
    pred = model(xb)
    pred = (nn.Sigmoid()(pred)>0.5).double().cpu().detach()
    xb = xb.cpu().detach()
    plot_batch(xb, pred, 5)


