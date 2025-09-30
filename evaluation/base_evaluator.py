import os
import time
import torch
import numpy as np
from models.cttp.cttp_model import CTTP
import yaml
import tqdm
import numpy as np
from scipy import linalg
import random

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

class BaseEvaluator:
    def __init__(self, configs, dataset, model):
        self._init_cfgs(configs)
        self._init_model(model)
        self._init_data(dataset)
        if "clip_config_path" in configs.keys():
            self._init_clip(configs)

    def _init_clip(self, configs):
        model_dict = {
            "clip_patchtst": CTTP,
        }
        clip_configs = yaml.safe_load(open(configs["clip_config_path"]))
        self.clip = model_dict[clip_configs["clip_type"]](clip_configs)
        self.clip.load_state_dict(torch.load(configs["clip_model_path"]))
        self.clip = self.clip.to(self.clip.device)

        fid_mean_cache_path = os.path.join(configs["cache_folder"], "fid_mean.npy")
        fid_cov_cache_path = os.path.join(configs["cache_folder"], "fid_cov.npy")
        jftsd_mean_cache_path = os.path.join(configs["cache_folder"], "jftsd_mean.npy")
        jftsd_cov_cache_path = os.path.join(configs["cache_folder"], "jftsd_cov.npy")
        print("cache_folder: ", configs["cache_folder"])
        if os.path.exists(fid_mean_cache_path) and os.path.exists(fid_cov_cache_path) and os.path.exists(jftsd_mean_cache_path) and os.path.exists(jftsd_cov_cache_path):
            self.ts_mean = np.load(fid_mean_cache_path)
            self.ts_cov = np.load(fid_cov_cache_path)
            self.joint_mean = np.load(jftsd_mean_cache_path)
            self.joint_cov = np.load(jftsd_cov_cache_path)
        else:
            train_loader = self.dataset.get_loader(split="train", batch_size=self.batch_size, shuffle=False, include_self=False)
            all_ts_emb, all_joint_emb = [], []
            with torch.no_grad():
                print("calc the ts mean and cov")
                for batch in tqdm.tqdm(train_loader):
                    ts = batch["ts"].to(self.clip.device).float()
                    ts_len = batch["ts_len"].to(self.clip.device).int()
                    cap = batch["cap"]
                    ts_emb = self.clip.get_ts_coemb(ts, ts_len)
                    cap_emb = self.clip.get_text_coemb(cap, None)
                    all_ts_emb.append(ts_emb)
                    all_joint_emb.append(torch.cat([ts_emb,cap_emb], dim=-1))

            all_ts_emb = torch.cat(all_ts_emb, dim=0)
            all_ts_emb = all_ts_emb.cpu().numpy()
            self.ts_mean = np.mean(all_ts_emb, axis=0)
            self.ts_cov = np.cov(all_ts_emb, rowvar=False)
            all_joint_emb = torch.cat(all_joint_emb, dim=0)
            all_joint_emb = all_joint_emb.cpu().numpy()
            self.joint_mean = np.mean(all_joint_emb, axis=0)
            self.joint_cov = np.cov(all_joint_emb, rowvar=False)

            os.makedirs(configs["cache_folder"], exist_ok=True)
            np.save(fid_mean_cache_path, self.ts_mean)
            np.save(fid_cov_cache_path, self.ts_cov)
            np.save(jftsd_mean_cache_path, self.joint_mean)
            np.save(jftsd_cov_cache_path, self.joint_cov)

    def _init_cfgs(self, configs):
        self.configs = configs
        self.batch_size = self.configs["batch_size"]
        self.n_samples = self.configs["n_samples"]
        self.display_epoch_interval = self.configs["display_interval"]
        self.model_path = self.configs["model_path"]

    def _init_model(self, model):
        self.model = model
        if self.model_path != "":
            print("Loading pretrained model from {}".format(self.model_path))
            self.model.load_state_dict(torch.load(self.model_path))

    def _init_data(self, dataset):
        self.dataset = dataset
        self.test_loader = dataset.get_loader(split="test", batch_size=self.batch_size, shuffle=False, include_self=False)

    """
    Evaluate.
    """
    def evaluate(self, mode="cond_gen", sampler="ddpm", save_pred=False):
        """
        Args:
            mode: cond_gen or edit.
            sampler: ddpm or ddim.
        """
        print("\n-------------------------------")
        print(f"Evaluating the model with mode={mode} and sampler={sampler}")
        self.model.eval()
        all_tsgen_emb = []
        all_joint_emb = []
        cttp = 0
        sample_num = 0

        with torch.no_grad():
            for batch_no, batch in enumerate(self.test_loader):
                start_time = time.time()
                multi_preds = self.model.generate(batch, self.n_samples, sampler)
                multi_preds = multi_preds.permute(0,1,3,2)
                pred = multi_preds.median(dim=0).values

                ts = batch["ts"].to(self.model.device).float()
                ts_len = batch["ts_len"].to(self.model.device).int()
                ts_gt_emb = self.clip.get_ts_coemb(ts, ts_len)
                cap_tokens = batch["cap"]
                cap_emb = self.clip.get_text_coemb(cap_tokens, None)

                if "clip_config_path" in self.configs.keys():
                    ts_gen_emb = self.clip.get_ts_coemb(pred, ts_len)
                    all_tsgen_emb.append(ts_gen_emb)
                    all_joint_emb.append(torch.cat([ts_gen_emb,cap_emb], dim=-1))
                    cttp += torch.mm(ts_gen_emb, cap_emb.permute(1,0)).trace().item()
                    sample_num += ts_gen_emb.shape[0]

                end_time = time.time()
                if (batch_no+1)%self.display_epoch_interval == 0:
                    print("Batch", batch_no, 
                        "Batch Time {:.2f}s".format(end_time-start_time))
        cttp /= sample_num
        print("Done!")
        res_dict = {
            "tensorboard":{},
            "df":{},
        }
        if "clip_config_path" in self.configs.keys():
            res_dict["tensorboard"].update({"cttp": cttp})
            res_dict["df"].update({"cttp": cttp})
            fid = None
            jftsd = None
            tsgen_emb = []
            joint_emb = []

            all_tsgen_emb = torch.cat(all_tsgen_emb, dim=0).cpu().numpy()
            tsgen_mean = np.mean(all_tsgen_emb, axis=0)
            tsgen_var = np.cov(all_tsgen_emb, rowvar=False)
            fid = calculate_frechet_distance(self.ts_mean, self.ts_cov, tsgen_mean, tsgen_var)
            all_joint_emb = torch.cat(all_joint_emb, dim=0).cpu().numpy()
            joint_mean = np.mean(all_joint_emb, axis=0)
            joint_var = np.cov(all_joint_emb, rowvar=False)
            jftsd = calculate_frechet_distance(self.joint_mean, self.joint_cov, joint_mean, joint_var)
            
            res_dict["tensorboard"].update({"fid":fid})
            res_dict["df"].update({"fid":fid})
            res_dict["tensorboard"].update({"jftsd":jftsd})
            res_dict["df"].update({"jftsd":jftsd})
            
            print("FID: ", fid)
            print("JFTSD: ", jftsd)
            print("CTTP ", cttp)

        return res_dict