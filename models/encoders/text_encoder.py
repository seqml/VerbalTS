import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, BertModel
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPTextConfig

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", batch_first=True
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

class CLIPTextEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.device = configs["device"]
        self.emb_dim = configs["text_emb"]
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
            nn.Linear(configs["textemb_hidden_dim"], configs["text_emb"])
        )
    def forward(self, text):
        inputs = self.tokenizer(text, padding=True, return_tensors="pt")["input_ids"]
        inputs = inputs.to(self.device)
        if "output_type" not in self.configs.keys():
            self.configs["output_type"] = "cls"
        if self.configs["output_type"] == "cls":
            text_emb = self.model(input_ids=inputs).text_embeds
            text_co_emb = self.text_enc(text_emb)
            text_co_emb = text_co_emb[:, None, :]
        if self.configs["output_type"] == "all":
            text_emb = self.model(input_ids=inputs).last_hidden_state
            text_co_emb = self.text_enc(text_emb)
            
        return text_co_emb

class TextEncoder(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs
        self.device = configs["device"]
        self.emb_dim = configs["text_emb"]
        self.vocab_emb = nn.Embedding(num_embeddings=configs["word_size"], embedding_dim=self.emb_dim)
        self.trans_layer = get_torch_trans(heads=8, layers=2, channels=64)
        self.tokenizer = AutoTokenizer.from_pretrained(configs["tokenizer_path"])
        if configs["pos_emb"] != "none":
            self.init_pe(self.emb_dim)
    
    def init_pe(self, d_model, max_len=5000):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
        self.pe.requires_grad = False

    def forward(self, text):
        inputs = self.tokenizer(text, padding=True, return_tensors="pt")["input_ids"]
        inputs = inputs.to(self.device)
        text_emb = self.vocab_emb(inputs)
        if self.configs["pos_emb"] != "none":
            text_emb += self.pe[:, :text_emb.shape[1], :].to(text_emb.device)
        text_emb = self.trans_layer(text_emb)
        return text_emb
