import torch
from torch import nn
import torch.nn.functional as F
import os
import numpy as np

from .patchtst_modules import PatchEmbedding, Encoder, EncoderLayer, AttentionLayer, FullAttention
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, BertModel
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPTextConfig
import math

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=3.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

class CLIPTextEncoder(nn.Module):
    def __init__(self, configs):
        super(CLIPTextEncoder, self).__init__()
        self.device = configs["device"]
        if "output_type" in configs.keys():
            self.output_type = configs["output_type"]
        else:
            self.output_type = "cls"
        if "Longclip" in configs["pretrain_model_path"]:
            clip_config = CLIPTextConfig.from_pretrained(configs["pretrain_model_path"])
            clip_config.max_position_embeddings = 248
            self.model = CLIPTextModelWithProjection.from_pretrained(configs["pretrain_model_path"], config=clip_config)
            self.tokenizer = AutoTokenizer.from_pretrained(configs["pretrain_model_path"])
        else:
            self.model = CLIPTextModelWithProjection.from_pretrained(configs["pretrain_model_path"])
            self.tokenizer = AutoTokenizer.from_pretrained(configs["pretrain_model_path"])
        self.max_length = self.model.config.max_position_embeddings

        for i, (name, param) in enumerate(self.model.named_parameters()):
            param.requires_grad = False

        self.text_enc = nn.Sequential(
            nn.Linear(configs["pretrain_model_dim"], configs["textemb_hidden_dim"]),
            nn.LayerNorm(configs["textemb_hidden_dim"]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(configs["textemb_hidden_dim"], configs["coemb_dim"])
        )
    def forward(self, text, text_len):
        inputs = self.tokenizer(text, padding=True, return_tensors="pt")["input_ids"]
        inputs = inputs.to(self.device)
        chunks = inputs.split(self.max_length, dim=1)
        if self.output_type == "cls":
            embeddings = [self.model(input_ids=chunk).text_embeds for chunk in chunks]
            text_emb = torch.mean(torch.stack(embeddings), dim=0)
        elif self.output_type == "all":
            embeddings = [torch.mean(self.model(input_ids=chunk).last_hidden_state, dim=1) for chunk in chunks]
            text_emb = torch.mean(torch.stack(embeddings), dim=0)
        text_co_emb = self.text_enc(text_emb)
        return text_co_emb

class CTTP(nn.Module):
    def __init__(self, configs):
        super(CTTP, self).__init__()
        configs["text"]["device"] = configs["device"]
        configs["ts"]["device"] = configs["device"]
        self.text_enc = CLIPTextEncoder(configs["text"])
        if configs["ts"]["type"] == "patchtst_mae_pretrain":
            self.ts_enc = PatchTST_MAE(configs["ts"])

        self.device = configs["device"]
        self.configs = configs
        self.CE = nn.CrossEntropyLoss(reduction="none")
        self.contrastive_loss = ContrastiveLoss()
        if "loss_type" in configs.keys():
            self.loss_type = configs["loss_type"]
        else:
            self.loss_type = "Contrastive"
    
    def forward(self, ts, ts_len, text, text_len):
        """
        ts: (B, T, C)
        text: (B, L, V)
        """
        B = ts.shape[0]

        ts_co_emb = self.ts_enc(ts, ts_len)
        text_co_emb = self.text_enc(text, text_len)
        
        loss_dict = {}

        if self.loss_type == "Contrastive":
            pos_labels = torch.zeros(B).to(ts_co_emb.device)
            loss_dict["positive"] = self.contrastive_loss(text_co_emb, ts_co_emb, pos_labels)
            '''Negative Pairs, shifting index'''
            neg_labels = torch.ones(B).to(ts_co_emb.device)
            shift = np.random.randint(0, B-1)
            new_idx = np.arange(shift, B + shift) % B
            mis_ts_co_emb = ts_co_emb.clone()[new_idx]
            loss_dict["negative"] = self.contrastive_loss(text_co_emb, mis_ts_co_emb, neg_labels)
            loss_dict["all"] = loss_dict["positive"] + loss_dict["negative"]

        elif self.loss_type == "CE":
            sim = torch.mm(ts_co_emb, text_co_emb.permute(1,0)) # (B,B)
            labels = torch.arange(sim.shape[0], device=sim.device)  # (B)
            loss_dict["ts2text"] = self.CE(torch.reshape(sim, [B,-1]), labels)
            sim = sim.permute(1,0)
            loss_dict["text2ts"] = self.CE(torch.reshape(sim, [B,-1]), labels)
            loss_dict["ts2text"] = torch.mean(loss_dict["ts2text"], dim=-1)
            loss_dict["text2ts"] = torch.mean(loss_dict["text2ts"], dim=-1)
            loss_dict["all"] = (loss_dict["text2ts"] + loss_dict["ts2text"]) / 2

        return loss_dict
    
    def retrive_cloest(self, ts, ts_len, text, text_len):
        ts_co_emb = self.ts_enc(ts, ts_len)
        text_co_emb = self.text_enc(text, text_len)
        sim = torch.nn.functional.softmax(torch.mm(ts_co_emb, text_co_emb.permute(1,0)), dim=-1)
        return sim
    
    def get_ts_coemb(self, ts, ts_len):
        ts_co_emb = self.ts_enc(ts, ts_len)
        return ts_co_emb
    
    def get_text_coemb(self, text, text_len):
        text_co_emb = self.text_enc(text, text_len)
        return text_co_emb

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", batch_first=True
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

class PatchTST_MAE(nn.Module):
    def __init__(self, configs):
        super(PatchTST_MAE, self).__init__()
        self.patch_encoder = PatchEncoder(configs=configs)
        if configs["pretrain_encoder_path"] != "":
            pretrain_encoder_path = configs["pretrain_encoder_path"]
            print(f"Load pretrain ts encoder from {pretrain_encoder_path}")
            self.patch_encoder.load_state_dict(torch.load(configs["pretrain_encoder_path"]))

        self.time_transformer = get_torch_trans(heads=configs["n_heads"], layers=1, channels=configs["d_model"])
        self.var_transformer = get_torch_trans(heads=configs["n_heads"], layers=1, channels=configs["d_model"])
        self.out_projector = nn.Linear(configs["d_model"], configs["coemb_dim"])
    
    def forward(self, ts, ts_len):
        B, L, N = ts.shape
        ts_var_emb = self.patch_encoder(ts) #[B*N, Nl, d_model]
        var_emb = self.time_transformer(ts_var_emb)[:,:1,:].reshape(B, N, -1) #[B, N, d_model]
        co_emb = self.var_transformer(var_emb)[:,:1,:].reshape(B, -1) #[B, d_model]
        co_emb = self.out_projector(co_emb)
        return co_emb

class PatchEncoder(nn.Module):
    def __init__(self, configs):
        super(PatchEncoder, self).__init__()
        self.device = configs["device"]
        self.patch_embedding = PatchEmbedding(configs["d_model"], configs["patch_len"], 1, configs["stride"], configs["padding"], configs["dropout"])
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs["factor"], attention_dropout=configs["dropout"],
                                      output_attention=configs["output_attention"]), configs["d_model"], configs["n_heads"]),
                    configs["d_model"],
                    configs["d_ff"],
                    dropout=configs["dropout"],
                    activation=configs["activation"]
                ) for l in range(configs["e_layers"])
            ],
            norm_layer=torch.nn.LayerNorm(configs["d_model"])
        )
        patch_seq_len = int((configs["seq_len"] - configs["patch_len"]) / configs["stride"] + 2)
        tp = torch.tensor([i for i in range(patch_seq_len)]).to(self.device)
        tp = tp[None]
        self.time_pos_emb = self.time_embedding(tp, d_model=configs["d_model"])
        self.time_pos_emb.requires_grad = False
        self.var_pos_emb = nn.Embedding(num_embeddings=configs["n_var"], embedding_dim=configs["d_model"]).to(self.device)
    
    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, ts):
        B, L, N = ts.shape
        ts = ts.permute(0, 2, 1).reshape(B*N, 1, L) # (B*N, L)
        ts_emb = self.patch_embedding(ts) # (B*N, Nl, d_model)
        BN, Nl, D = ts_emb.shape
        timposemb = self.time_pos_emb.expand((BN,-1,-1))
        varposemb = self.var_pos_emb(torch.arange(N).to(self.device))[None].expand(B,-1,-1) # (B, N, d_model)
        varposemb = varposemb.reshape(B*N, 1, -1) # (B*N, 1, d_model)
        ts_emb += timposemb + varposemb
        ts_enc_out, attns = self.encoder(ts_emb) # (B*N, Nl, d_model)
        return ts_enc_out

    def mask_forward(self, ts, mask_ratio):
        B, L, N = ts.shape
        ts = ts.permute(0, 2, 1).reshape(B*N, 1, L) # (B*N, L)
        ts_emb = self.patch_embedding(ts) # (B*N, Nl, d_model)
        BN, Nl, D = ts_emb.shape
        zero_indices = torch.multinomial(torch.ones(BN, Nl), int(Nl*mask_ratio), replacement=False)  # [BN, Nl]
        mask = torch.ones(BN, Nl, dtype=torch.int)
        batch_indices = torch.arange(BN).unsqueeze(1)  # [BN, 1]
        mask[batch_indices, zero_indices] = 0 # [BN, Nl]
        mask = mask[:,:,None].to(self.device) # [BN, Nl, 1]
        ts_emb_mask = ts_emb * mask 

        timposemb = self.time_pos_emb.expand((BN,-1,-1))
        varposemb = self.var_pos_emb(torch.arange(N).to(self.device))[None].expand(B,-1,-1) # (B, N, d_model)
        varposemb = varposemb.reshape(B*N, 1, -1) # (B*N, 1, d_model)
        ts_emb += timposemb + varposemb

        ts_enc_out, attns = self.encoder(ts_emb_mask)
        return ts_enc_out