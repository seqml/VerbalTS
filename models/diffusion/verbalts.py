import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
import copy
from models.encoders.side_encoder import SideEncoder_Var

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", batch_first=True
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def get_torch_cross_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerDecoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", batch_first=True
    )
    return nn.TransformerDecoder(encoder_layer, num_layers=layers)

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

class TsPatchEmbedding(nn.Module):
    def __init__(self, L_patch_len, channels, d_model, dropout):
        super(TsPatchEmbedding, self).__init__()
        self.L_patch_len = L_patch_len
        self.padding_patch_layer = nn.ReplicationPad2d((0, L_patch_len, 0, 0))
        self.value_embedding = nn.Sequential(
            nn.Linear(L_patch_len*channels, d_model),
            nn.ReLU(),
        )

    def forward(self, x_in):
        if x_in.shape[-1] % self.L_patch_len:
            x = self.padding_patch_layer(x_in)
        else:
            x = x_in
        x = x.unfold(dimension=3, size=self.L_patch_len, step=self.L_patch_len)
        B, C, n_var, Nl, Pl = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous().reshape(B, n_var, Nl, Pl*C)
        x = self.value_embedding(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class SidePatchEmbedding(nn.Module):
    def __init__(self, L_patch_len, channels, d_model, dropout):
        super(SidePatchEmbedding, self).__init__()
        self.L_patch_len = L_patch_len
        self.padding_patch_layer = nn.ReplicationPad2d((0, L_patch_len, 0, 0))
        self.value_embedding = nn.Sequential(
            nn.Linear(L_patch_len*channels, d_model),
        )

    def forward(self, x_in):
        if x_in.shape[-1] % self.L_patch_len:
            x = self.padding_patch_layer(x_in)
        else:
            x = x_in
        x = x.unfold(dimension=3, size=self.L_patch_len, step=self.L_patch_len)
        B, C, n_var, Nl, Pl = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous().reshape(B, n_var, Nl, Pl*C)
        x = self.value_embedding(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class PatchDecoder(nn.Module):
    def __init__(self, L_patch_len, d_model, channels):
        super().__init__()
        self.L_patch_len = L_patch_len
        self.channels = channels
        self.linear = nn.Linear(d_model, L_patch_len*channels)

    def forward(self, x):
        B, D, n_var, Nl = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.linear(x)
        x = x.reshape(B, n_var, Nl, self.L_patch_len, self.channels).permute(0, 4, 1, 2, 3).contiguous()
        x = x.reshape(B, self.channels, n_var, Nl*self.L_patch_len)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, condition_type="add"):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.side_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        if condition_type == "add":
            pass
        elif condition_type == "cross_attention":
            self.condition_cross_attention = get_torch_cross_trans(heads=nheads, layers=1, channels=channels)
        elif condition_type == "adaLN":
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 3 * channels, bias=True)
            )

    def forward_time(self, y, base_shape, attention_mask=None):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(0, 2, 1), mask=attention_mask).permute(0, 2, 1)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape, attention_mask=None):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(0, 2, 1), mask=attention_mask).permute(0, 2, 1)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y
    
    def foward_cross_attention(self, y, cond, attetion_mask=None):
        B, channel, K, L = y.shape
        y = y.reshape(B, channel, K, L).permute(0, 2, 3, 1).reshape(B * K, L, channel)
        cond = cond.reshape(B, channel, K, L).permute(0, 2, 3, 1).reshape(B * K, L, channel)
        y = self.condition_cross_attention(tgt=y, memory=cond, memory_mask=attetion_mask).permute(0, 2, 1)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3)
        return y

    def modulate(self, x, shift, scale):
        return x * (1 + scale) + shift

    def forward(self, x, side_emb, attr_emb, diffusion_emb, attention_mask=None, condition_type="add"):
    
        if condition_type == "add":
            x = x + attr_emb
        elif condition_type == "cross_attention":
            x = self.foward_cross_attention(x, attr_emb, attetion_mask=attention_mask)
        elif condition_type == "adaLN":
            gama, beta, alpha = self.adaLN_modulation(attr_emb.permute(0,2,3,1)).chunk(3, dim=-1)
            gama, beta, alpha = gama.permute(0,3,1,2), beta.permute(0,3,1,2), alpha.permute(0,3,1,2)

        B, channel, K, L = x.shape
        base_shape = x.shape

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1).unsqueeze(-1)
        y = x + diffusion_emb
        if condition_type == "adaLN":
            y = self.modulate(y, gama, beta)
        
        y = self.forward_time(y, base_shape, attention_mask)
        y = self.forward_feature(y, base_shape, None)

        if condition_type == "adaLN":
            y = y.reshape(B,channel,K,L)
            y = alpha * y
            y = y.reshape(B,channel,K*L)

        y = y.reshape(B,channel,K*L)
        y = self.mid_projection(y)

        _, side_dim, _, _ = side_emb.shape
        side_emb = side_emb.reshape(B, side_dim, K * L)
        side_emb = self.side_projection(side_emb)
        y = y + side_emb

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip

class VerbalTS(nn.Module):
    def __init__(self, config, inputdim=1):
        super().__init__()
        self.config = config
        self.n_var = config["n_var"]
        self.var_dep_type = config["var_dep_type"]
        self.channels = config["channels"]
        self.multipatch_num = config["multipatch_num"]
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        config["side"]["device"] = config["device"]
        self.side_encoder = SideEncoder_Var(configs=config["side"])
        side_dim = self.side_encoder.total_emb_dim

        self.attention_mask_type = config["attention_mask_type"]
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.ts_downsample = nn.ModuleList([])
        self.side_downsample = nn.ModuleList([])
        self.patch_decoder = nn.ModuleList([])

        for i in range(self.multipatch_num):
            self.ts_downsample.append(TsPatchEmbedding(L_patch_len=config["base_patch"]*config["L_patch_len"]**i, channels=inputdim, d_model=self.channels, dropout=0))
            self.patch_decoder.append(PatchDecoder(L_patch_len=config["base_patch"]*config["L_patch_len"]**i, d_model=self.channels, channels=1))
            self.side_downsample.append(SidePatchEmbedding(L_patch_len=config["base_patch"]*config["L_patch_len"]**i, channels=side_dim, d_model=side_dim, dropout=0))
        self.multipatch_mixer = nn.Linear(self.multipatch_num, 1)
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=side_dim,
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    condition_type=config["condition_type"]
                )
                for _ in range(config["layers"])
            ]
        )
        
    def forward(self, x_raw, tp, attr_emb_raw, diffusion_step):
        B_raw, inputdim, n_var, L = x_raw.shape
        side_emb_raw = self.side_encoder(tp)
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        
        x_list = []
        side_list = []
        scale_length = []
        for i in range(self.multipatch_num):
            x = self.ts_downsample[i](x_raw)
            side_emb = self.side_downsample[i](side_emb_raw)
            x_list.append(x)
            side_list.append(side_emb)
            scale_length.append(x.shape[-1])

        if self.attention_mask_type == "full" or attr_emb_raw is None:
            attention_mask = None
        elif self.attention_mask_type == "parallel":
            attention_mask = self.get_mask(0, [x_list[i].shape[-1] for i in range(len(x_list))], device=x_raw.device)
        
        x_in = torch.cat(x_list, dim=-1)
        side_in = torch.cat(side_list, dim=-1)

        if attr_emb_raw is None:
            attr_emb = torch.zeros_like(x_in)
        else:
            if "scale" in self.config["text_projector"]:
                assert len(scale_length) == attr_emb_raw.shape[2]
                mscale_attr_list = []
                for i in range(len(scale_length)):
                    tmp_scale_attr = attr_emb_raw[:,:,i:i+1,:].expand([-1, -1, scale_length[i], -1])
                    mscale_attr_list.append(tmp_scale_attr)
                attr_emb = torch.cat(mscale_attr_list, dim=2)
            else:
                attr_emb = attr_emb_raw.expand([-1, -1, x_in.shape[-1], -1])
            attr_emb = attr_emb.permute(0,3,1,2)

        B, _, Nk, Nl = x_in.shape
        _x_in = x_in
        skip = []
        for layer in self.residual_layers:
            x_in, skip_connection = layer(x_in+_x_in, side_in, attr_emb, diffusion_emb, attention_mask=attention_mask, condition_type=self.config["condition_type"])
            skip.append(skip_connection)
        
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))

        x = x.reshape(B, self.channels, Nk * Nl)
        x = self.output_projection1(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, Nk, Nl)

        start_id = 0
        all_out = []
        for i in range(len(x_list)):
            x_out = x[:,:,:,start_id:start_id+x_list[i].shape[-1]]
            x_out = self.patch_decoder[i](x_out)
            x_out = x_out[:, :, :, :L]
            all_out.append(x_out)
            start_id += x_list[i].shape[-1]

        all_out = torch.cat(all_out, dim=1)
        all_out = self.multipatch_mixer(all_out.permute(0, 2, 3, 1).contiguous())
        all_out = all_out.reshape((B_raw, n_var, L))
        return all_out, {}
    
    def get_mask(self, attr_len, len_list, device="cuda:0"):
        total_len = sum(len_list) + attr_len
        mask = torch.zeros(total_len, total_len, device=device) - float("inf")
        mask[:attr_len, :] = 0
        mask[:, :attr_len] = 0
        start_id = attr_len
        for i in range(len(len_list)):
            mask[start_id:start_id+len_list[i], start_id:start_id+len_list[i]] = 0
            start_id += len_list[i]
        return mask