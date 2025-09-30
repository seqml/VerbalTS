import torch
import torch.nn as nn

from models.encoders.attr_encoder import AttributeEncoder
from models.encoders.text_encoder import TextEncoder, CLIPTextEncoder
from models.encoders.cond_projector import TextProjectorMVarMScaleMStep, AttrProjectorAvg
from models.unconditional_generator import UnConditionalGenerator
from models.cttp.cttp_model import CTTP
import time
import random
import yaml

class ConditionalGenerator(nn.Module):
    def __init__(self, diff_configs, cond_configs):
        super().__init__()
        self.device = diff_configs["device"]
        self.diff_configs = diff_configs
        self.cond_configs = cond_configs
        self._init_condition_encoders(diff_configs, cond_configs)
        self._init_diff(diff_configs)

    def _init_condition_encoders(self, diff_configs, cond_configs):
        if cond_configs["cond_modal"] == "constraint":
            clip_configs = yaml.safe_load(open(cond_configs["constraint"]["clip_config_path"]))
            self.cond_guide_model = CTTP(clip_configs)
            self.cond_guide_model.load_state_dict(torch.load(cond_configs["constraint"]["clip_model_path"]))
            self.cond_guide_model = self.cond_guide_model.to(self.cond_guide_model.device)
        elif cond_configs["cond_modal"] == "attr":
            cond_configs["attrs"]["device"] = self.device
            self.attr_en = AttributeEncoder(cond_configs["attrs"]).to(self.device)
            self.cond_projector = AttrProjectorAvg(dim_in=cond_configs["attrs"]["attr_emb"], dim_hid=diff_configs["diffusion"]["channels"], dim_out=diff_configs["diffusion"]["channels"])
            self.cond_projector = self.cond_projector.to(self.device)
        elif "text" in cond_configs["cond_modal"]:
            if cond_configs["cond_modal"] == "text":
                cond_configs["text"]["device"] = self.device
                self.attr_en = CLIPTextEncoder(cond_configs["text"]).to(self.device)
            elif cond_configs["cond_modal"] == "simple_text":
                cond_configs["text"]["device"] = self.device
                self.attr_en = TextEncoder(cond_configs["text"]).to(self.device)
            if cond_configs["text"]["text_projector"] == "var_scale_diffstep_multi":
                self.cond_projector = TextProjectorMVarMScaleMStep(n_var=diff_configs["diffusion"]["n_var"],
                                                         n_scale=diff_configs["diffusion"]["multipatch_num"],
                                                         n_steps=diff_configs["diffusion"]["num_steps"],
                                                         n_stages=cond_configs["text"]["num_stages"],
                                                         dim_in=cond_configs["text"]["text_emb"], 
                                                         dim_out=diff_configs["diffusion"]["channels"])
            self.cond_projector = self.cond_projector.to(self.device)

    def _init_diff(self, configs):
        configs["device"] = self.device
        if "text" in self.cond_configs["cond_modal"]:
            configs["diffusion"]["text_projector"] = self.cond_configs["text"]["text_projector"]
        self.generator = UnConditionalGenerator(configs=configs)
        if configs["generator_pretrain_path"] != "":
            self.generator.load_state_dict(torch.load(configs["generator_pretrain_path"]))
            print("Load the pretrain unconditional generator")
        else:
            print("Learn from scratch")
    
    """
    Finetune.
    """
    def forward(self, batch, is_train):
        x, tp, attrs = self._unpack_data_cond_gen(batch)
        attr_emb_raw = self.attr_en(attrs)
        if self.cond_configs["cond_modal"] == "attr" or "diffstep" not in self.cond_configs["text"]["text_projector"]:
            attr_emb = self.cond_projector(attr_emb_raw)

        B = x.shape[0]
        if is_train:
            t = torch.randint(0, self.generator.num_steps, [B], device=self.device)
            if "text" in self.cond_configs["cond_modal"] and "diffstep" in self.cond_configs["text"]["text_projector"]:
                attr_emb = self.cond_projector(attr_emb_raw, t)
            loss = self.generator._noise_estimation_loss(x, tp, attr_emb, t)
            return loss
        
        loss_dict = {}
        for t in range(self.generator.num_steps):
            t = (torch.ones(B, device=self.device) * t).long()
            if "text" in self.cond_configs["cond_modal"] and "diffstep" in self.cond_configs["text"]["text_projector"]:
                attr_emb = self.cond_projector(attr_emb_raw, t)
            tmp_loss_dict = self.generator._noise_estimation_loss(x, tp, attr_emb, t)
            for k in tmp_loss_dict:
                if k in loss_dict.keys():
                    loss_dict[k] += tmp_loss_dict[k]
                else:
                    loss_dict[k] = tmp_loss_dict[k]
        for k in loss_dict:
            loss_dict[k] = loss_dict[k] / self.generator.num_steps
        return loss_dict

    def _unpack_data_cond_gen(self, batch):
        ts = batch["ts"].to(self.device).float()
        tp = batch["tp"].to(self.device).float()
        if "text" in self.cond_configs["cond_modal"]:
            attrs = batch["cap"]
        elif "constraint" in self.cond_configs["cond_modal"]:
            attrs = batch["cap"]
        elif self.cond_configs["cond_modal"] == "attr":
            attrs = batch["attrs"].to(self.device).long()
        ts = ts.permute(0, 2, 1)
        return ts, tp, attrs

    def generate(self, batch, n_samples, sampler="ddim"):
        if self.cond_configs["cond_modal"] == "constraint":
            return self.generate_constraint(batch, n_samples, sampler)
        else:
            return self.generate_text(batch, n_samples, sampler)

    """
    Generation.
    """
    @torch.no_grad()
    def generate_text(self, batch, n_samples, sampler="ddim"):
        ts, tp, attrs = self._unpack_data_cond_gen(batch)
        attr_emb_raw = self.attr_en(attrs)
        if self.cond_configs["cond_modal"] == "attr" or "diffstep" not in self.cond_configs["text"]["text_projector"]:
            attr_emb = self.cond_projector(attr_emb_raw)

        samples = []
        B = ts.shape[0]
        for i in range(n_samples):
            x = torch.randn_like(ts)
            for t in range(self.generator.num_steps-1, -1, -1):
                noise = torch.randn_like(x)
                t = (torch.ones(B, device=self.device) * t).long()
                if "text" in self.cond_configs["cond_modal"] and "diffstep" in self.cond_configs["text"]["text_projector"]:
                    attr_emb = self.cond_projector(attr_emb_raw, t)
                pred_noise, _ = self.generator.predict_noise(x, tp, attr_emb, t)
                if sampler == "ddpm":
                    x = self.generator.ddpm.reverse(x, pred_noise, t, noise)
                else:
                    x = self.generator.ddim.reverse(x, pred_noise, t, noise, is_determin=True)
            samples.append(x)
        return torch.stack(samples)
    
    def generate_constraint(self, batch, n_samples, sampler="ddim"):
        ts, tp, attrs = self._unpack_data_cond_gen(batch)
        samples = []
        B = ts.shape[0]
        for i in range(n_samples):
            x = torch.randn_like(ts)
            for t in range(self.generator.num_steps-1, -1, -1):
                noise = torch.randn_like(x)
                t = (torch.ones(B, device=self.device) * t).long()
                with torch.no_grad():
                    pred_noise, _ = self.generator.predict_noise(x, tp, None, t)
                if sampler == "ddpm":
                    x = self.generator.ddpm.reverse(x, pred_noise, t, noise)
                else:
                    x0 = self.generator.ddim.predict_x0(x, pred_noise, t).permute(0,2,1)
                    with torch.set_grad_enabled(True):
                        x0.requires_grad = True
                        ts_emb = self.cond_guide_model.get_ts_coemb(x0, None)
                        text_emb = self.cond_guide_model.get_text_coemb(attrs, None)
                        negative_cttp = -torch.mm(ts_emb, text_emb.permute(1,0)).trace()
                        negative_cttp.backward()
                    pred_noise -= self.cond_configs["constraint"]["guide_w"] * self.generator.ddim.one_minus_alpha_bar_sqrt[t] * x0.grad.permute(0,2,1)
                    x = self.generator.ddim.reverse(x, pred_noise, t, noise, is_determin=True)
            samples.append(x)
        return torch.stack(samples)