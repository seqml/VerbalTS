import os
import json
import numpy as np
import random
from torch.utils.data import Dataset
import time

class CustomDataset:
    def __init__(self, folder, **kwargs):
        super().__init__()
        self.folder = folder
        self._load_meta()

    def _load_meta(self):
        self.meta = json.load(open(os.path.join(self.folder, "meta.json")))
        self.attr_list = self.meta["attr_list"]
        n_attr = len(self.attr_list)
        self.attr_ids = np.arange(n_attr)
        self.attr_n_ops = np.array(self.meta["attr_n_ops"])

    def get_split(self, split, *args):
        return CustomSplit(self.folder, split)


class CustomSplit(Dataset):
    def __init__(self, folder, split="train"):
        super().__init__()
        assert split in ("train", "valid", "test"), "Please specify a valid split."
        self.split = split            
        self.folder = folder
        self._load_data()

        print(f"Split: {self.split}, total samples {self.n_samples}.")

    def _load_data(self):
        ts = np.load(os.path.join(self.folder, self.split+"_ts.npy"))     # [n_samples, n_steps]
        attrs = np.load(os.path.join(self.folder, self.split+"_attrs_idx.npy"))  # [n_samples, n_attrs]
        caps = np.load(os.path.join(self.folder, self.split+fr"_text_caps.npy"), allow_pickle=True)
        self.ts, self.attrs, self.caps = ts, attrs, caps
        self.n_samples = self.ts.shape[0]
        self.n_steps = self.ts.shape[1]
        self.n_attrs = self.attrs.shape[1]
        self.time_point = np.arange(self.n_steps)

    def __getitem__(self, idx):
        cap_id = random.randint(0, len(self.caps[idx])-1)
        tmp_ts = self.ts[idx]
        if len(tmp_ts.shape) == 1:
            tmp_ts = tmp_ts[...,np.newaxis]
        return {"ts": tmp_ts,
                "ts_len": tmp_ts.shape[0],
                "attrs": self.attrs[idx],
                "cap": self.caps[idx][cap_id],
                "tp": self.time_point}

    def __len__(self):
        return self.n_samples