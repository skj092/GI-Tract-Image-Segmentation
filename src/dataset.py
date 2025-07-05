from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import albumentations as A
import torch


# dataset and dataloader
class BuildDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.image_paths = df['image_path'].tolist()
        self.mask_paths = df['mask_path'].tolist()
        self.transform = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        img = cv2.imread(img_path)
        img = img.astype('float32')
        img = img/np.max(img)
        mask = np.load(mask_path)
        mask = mask.astype('float32')
        mask = mask/255
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        img = img.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
        return torch.tensor(img), torch.tensor(mask)


def prepare_loaders(df, fold, data_transforms, CFG, debug=False):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)
    if debug:
        train_df = train_df.query("empty==0").head(32 * 3)
        valid_df = valid_df.query("empty==0").head(32 * 3)
    train_dataset = BuildDataset(train_df, transforms=data_transforms['train'])
    valid_dataset = BuildDataset(valid_df, transforms=data_transforms['valid'])

    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs if not debug else 20,
                              num_workers=4, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs if not debug else 20,
                              num_workers=4, shuffle=False, pin_memory=True)

    return train_loader, valid_loader

