# VerbalTS: The implementation codes of "VerbalTS: Generating Time Series from Texts" 📈
<div align="center">

[![project page](https://img.shields.io/badge/Project%20page-VerbalTS%20-lightblue)](https://seqml.github.io/VerbalTS/)&nbsp;
[![paper link](https://img.shields.io/badge/ICML-45631-b31b1b.svg)](https://icml.cc/virtual/2025/poster/45631)&nbsp;

</div>

<p align="center" style="font-size: larger;">
  <a href="https://icml.cc/virtual/2025/poster/45631">VerbalTS: Generating Time Series from Texts</a>
</p>

<div>
  <p align="center" style="font-size: larger;">
    <strong>ICML 2025</strong>
  </p>
</div>

<p align="center">
<img src="https://github.com/seqml/VerbalTS/blob/main/asset/verbalts.png" width=95%>
<p>
<be>

## Contribution
### 1. Model Architecture
We propose VerbalTS, which consists of two key components: a multi-view noise estimator and a multi-focal text processor.
<p align="center">
<img src="https://github.com/seqml/VerbalTS/blob/main/asset/pipeline.png" width=95%>
<p>
Our model considers the time series generation process from three perspectives: temporal view, spatial view, and diffusion view. The textual description is processed through multi-focal reprogramming, which integrates the relevant tokens through learnable anchor vectors. Finally, a condition adapter is applied to align the multi-semantic information from the text across the three views with the corresponding components of the time series. With the method above, we achieve fine-grained time series generation from the textual descriptions. 

### 2. Experimental Results
We compare our method, VerbalTS, with the baselines on two synthetic datasets Synth-M, Synth-U, two real-world datasets Weather, BlindWays, and two real-world augmented datasets ETTm1, Traffic. As shown in the table below, our method significantly improves the fidelity and semantic alignment of the generated time series.
<p align="center">
<img src="https://github.com/seqml/VerbalTS/blob/main/asset/main_exp.png" width=95%>
<p>

### 3. Demo
Our method supports using verbal language to generate or edit the time series.

https://github.com/user-attachments/assets/00f35cb7-d68b-4a25-9603-be0fcb3e4737

## Installation
### 1. Environment
```
torch==2.2.1
pandas==2.0.3
pyyaml==6.0.2
linear_attention_transformer==0.19.1
tensorboard==2.14.0
scikit-learn==1.3.2
```
You can use the following command to prepare your environment.
```
pip install -r requirements.txt
```
### 2. Dataset
Download the datasets from [Google Drive](https://drive.google.com/drive/folders/1N0zxkLdvpdjkwayKA2OZIJYP4nfzhOeF?usp=drive_link).
<details>
    <summary> Assume the datasets are in `/path/to/data/`. It should be like:</summary>
  
    /path/to/data/:
        synthetic_m/:
            meta.json
            train_ts.npy
            train_attrs_idx.npy
            train_caps.npy
            valid_ts.npy
            valid_attrs_idx.npy
            train_caps.npy
            ...
        Weather/:
            ...
   **NOTE: The arg `--data_folder=/path/to/data/` should be passed to the training script.**
</details>

### 3. Pretrained model checkpoints
Download the [LongCLIP](https://huggingface.co/zer0int/LongCLIP-GmP-ViT-L-14) from Huggingface, and put the model weights in `/path/to/save/`.

Download the checkpoints from [Google Drive](https://drive.google.com/drive/folders/17zQJlxj5j7eWr636vmYdw1sGqi-uW1i4?usp=drive_link).
<details>
    <summary> Assume the checkpoints are in `/path/to/save/`. It should be like:</summary>

    /path/to/save/:
        [dataset_name]_cttp:
            ...
        [dataset_name]_eval:
            [run_id]:
                ckpts:
                    model_best.pth
                train_configs.yaml
                eval_configs.yaml
                model_cond_configs.yaml
                model_diff_configs.yaml
            ...
        ...
    
  **NOTE: The arg `--save_folder=/path/to/save/` should be passed to the training script.**
</details>
   
## Training
### 1. Train scripts
To pretrain the model on the specific dataset.
```
bash scripts/dataset_name/train.sh
```
### 2. Results
After the training, check the results at the following path.
```
{save_folder}/{run_id}/results_stat.csv
{save_folder}/{run_id}/results_stat_condgen.csv
```
### 3. Evaluate with checkpoints
To evaluate the model with the checkpoints.
```
bash scripts/dataset_name/eval.sh
```
### 5. Device
All codes in this repository run on GPU by default. If you need to run on the CPU, please modify the device-related parameters in the config file.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If our work helps you in research, please give us a star or cite us using the following:
```
@article{gu2025verbalts,
  title={VerbalTS: Generating Time Series from Texts},
  author={Gu, Shuqi and Li, Chuyue and Jing, Baoyu and Ren, Kan},
  journal={International Conference on Machine Learning},
  year={2025}
}
```
