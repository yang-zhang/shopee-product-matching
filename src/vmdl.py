from utils import *

from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import transforms

BS = 256


class VDataset(Dataset):
    def __init__(self, df, p_imgs, transforms):
        self.df = df
        self.transforms = transforms
        self.p_imgs = p_imgs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.image[idx]
        img_path = f"{self.p_imgs}/{img_id}"
        img = Image.open(img_path)
        img = self.transforms(img)
        return img


# https://github.com/lukemelas/EfficientNet-PyTorch
tfms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def mk_dl(df, p_imgs):
    ds = VDataset(df, p_imgs, tfms)
    dl = DataLoader(
        dataset=ds,
        batch_size=BS,
        num_workers=NWKRS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )
    return dl


def mk_feats(df, p_imgs, mdl):
    dl = mk_dl(df, p_imgs)
    device = torch.device(DEVICE)
    mdl = mdl.to(device)
    mdl.eval()
    feats = np.zeros((len(df), 1280))
    i = 0
    for dat in dl:
        with torch.no_grad():
            fts = mdl.extract_features(dat.to(device)).mean(dim=(-1, -2))
        l = len(fts)
        feats[i : i + l, :] = fts.cpu().detach().numpy()
        i += l
    return feats


def mk_sims(df, p_imgs, mdl=None):
    if mdl is None:
        mdl = EfficientNet.from_pretrained("efficientnet-b0")
    feats = mk_feats(df, p_imgs, mdl)
    sims = cosine_similarity(feats)
    return sims
