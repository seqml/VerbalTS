import torch
import torch.nn as nn
import numpy as np


class AttributeEncoder(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.device = configs["device"]

        self.emb_dim = configs["attr_emb"]
        self.attr_emb, self.attr_shift = self._init_embs(configs["num_attr_ops"])
        self.n_attr = len(configs["num_attr_ops"])
        self.n_ops_list = configs["num_attr_ops"]

        # emb for empty token
        self.empty_emb = nn.Embedding(num_embeddings=1, embedding_dim=self.emb_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.GELU(),
            nn.Linear(self.emb_dim, self.emb_dim)
        )


    def _init_embs(self, n_ops):
        shift = np.cumsum(n_ops)  # the number of options for each attr
        shift = np.insert(shift, 0, 0)
        emb = nn.Embedding(num_embeddings=shift[-1], embedding_dim=self.emb_dim)
        shift = torch.from_numpy(shift[:-1]).unsqueeze(0).to(self.device) # (1, total_attrs)
        return emb, shift

    def forward(self, attrs, replace_with_empty=False):
        """
        Args:
            attrs: (B,K)
            replace_with_empty: whether use embeddings of the empty token.
        """
        if replace_with_empty:
            idx = torch.zeros(attrs.shape).long().to(self.device)
            emb = self.empty_emb(idx)
        else:
            emb = self.attr_emb(attrs+self.attr_shift)  # (B,N,d)
        emb = self.out_proj(emb)
        return emb

    def get_all_embs(self):
        emb = self.attr_emb.weight.data  # (N1+N2+...,d)
        emb = self.out_proj(emb)

        emb_list = []
        for i in range(self.n_attr-1):
            attr_embs = emb[self.attr_shift[0,i]:self.attr_shift[0,i+1]]
            emb_list.append(attr_embs)
        attr_embs = emb[self.attr_shift[0,-1]:]
        emb_list.append(attr_embs)
        return emb_list
