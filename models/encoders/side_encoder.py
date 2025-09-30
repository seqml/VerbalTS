import torch
import torch.nn as nn
    
class SideEncoder_Var(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs
        self.device = configs["device"]

        self.num_var = configs["num_var"]
        self.var_emb = nn.Embedding(num_embeddings=self.num_var, embedding_dim=configs["var_emb"]).to(self.device)
        self.var_ids = torch.arange(self.num_var).to(self.device)

        self.total_emb_dim = configs["var_emb"] + configs["time_emb"]

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, tp):
        B, L = tp.shape
        time_emb = self.time_embedding(tp, self.configs["time_emb"])
        time_emb = time_emb.unsqueeze(2).expand(-1, -1, self.num_var, -1)
        var_emb = self.var_emb(self.var_ids)
        var_emb = var_emb.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_emb = torch.cat([time_emb, var_emb], dim=-1)
        side_emb = side_emb.permute(0, 3, 2, 1)
        return side_emb