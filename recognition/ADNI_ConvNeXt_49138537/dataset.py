import os, json, random
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

def zscore(x):
    x = x.astype(np.float32)
    m = np.mean(x)
    s = np.std(x) + 1e-8
    return (x - m) / s

def resize_2d(x, out_hw=(224, 224)):
    t = torch.from_numpy(x).float()[None, None, ...]
    t = F.interpolate(t, size=out_hw, mode='bilinear', align_corners=False)
    return t[0, 0].numpy()

class ADNI2p5DTrainSlices(Dataset):
    def __init__(self, root, json_name, subject_ids, use_key='masked', num_groups=8, seed=42, augment=True):
        self.root = root
        self.use_key = use_key
        self.num_groups = max(1, int(num_groups))
        self.augment = augment
        self.rng = random.Random(seed)

        with open(os.path.join(root, json_name), 'r') as f:
            meta = json.load(f)
        self.label_map = {0: 1, 2: 0}
        self.records = []
        for sid in subject_ids:
            e = meta[sid]
            if int(e['label']) not in (0, 2):
                continue
            self.records.append({
                'sid': sid,
                'path': os.path.join(root, e[self.use_key]),
                'label': self.label_map[int(e['label'])]
            })

    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.records) * self.num_groups

    def _aug(self, s):
        if not self.augment:
            return s
        if self.rng.random() < 0.5:
            s = np.flip(s, axis=1).copy()
        s = s + np.random.normal(0, 0.01, s.shape).astype(np.float32)
        return s

    def __getitem__(self, idx):
        rec = self.records[idx // self.num_groups]
        vol = nib.load(rec['path']).get_fdata().astype(np.float32)
        vol = zscore(vol)
        H, W, D = vol.shape
        z = self.rng.randint(1, D - 2)
        s1, s2, s3 = vol[..., z - 1], vol[..., z], vol[..., z + 1]
        s1, s2, s3 = self._aug(s1), self._aug(s2), self._aug(s3)
        s1, s2, s3 = resize_2d(s1), resize_2d(s2), resize_2d(s3)
        img = np.stack([s1, s2, s3], axis=-1)
        x = self.tf(img)
        y = torch.tensor(rec['label']).long()
        return x, y, rec['sid']

class ADNI2p5DEvalSubjects(Dataset):
    def __init__(self, root, json_name, subject_ids, use_key='masked', groups=12, seed=123):
        self.root = root
        self.use_key = use_key
        self.groups = max(1, int(groups))
        self.rng = random.Random(seed)

        with open(os.path.join(root, json_name), 'r') as f:
            meta = json.load(f)
        self.label_map = {0: 1, 2: 0}
        self.items = []
        for sid in subject_ids:
            e = meta[sid]
            if int(e['label']) not in (0, 2):
                continue
            self.items.append({
                'sid': sid,
                'path': os.path.join(root, e[self.use_key]),
                'label': self.label_map[int(e['label'])]
            })

    def __len__(self):
        return len(self.items)

    def _one_group(self, vol):
        H, W, D = vol.shape
        z = self.rng.randint(1, D - 2)
        s1, s2, s3 = vol[..., z - 1], vol[..., z], vol[..., z + 1]
        s1, s2, s3 = resize_2d(zscore(s1)), resize_2d(zscore(s2)), resize_2d(zscore(s3))
        img = np.stack([s1, s2, s3], axis=0)
        return torch.from_numpy(img).float()

    def __getitem__(self, idx):
        rec = self.items[idx]
        vol = nib.load(rec['path']).get_fdata().astype(np.float32)
        xs = [self._one_group(vol) for _ in range(self.groups)]
        x = torch.stack(xs, dim=0)
        y = torch.tensor(rec['label']).long()
        return x, y, rec['sid']
