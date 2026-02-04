<h1 align="center">CoGenCast: A Coupled Autoregressive-Flow Generative Framework for Time Series Forecasting</h1>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.2.2" />
  <img src="https://img.shields.io/badge/Time%20Series-Forecasting-2F4F4F" alt="Time Series Forecasting" />
</p>

## 📝 Abstract
In this work, we propose CoGenCast, a hybrid generative framework that couples pre-trained LLMs with flow-matching mechanism for effective time series forecasting. This repository contains the official code for our [paper](https://arxiv.org/abs/2602.03564).

<details>
  <summary><b>Full Abstract</b></summary>

  Time series forecasting can be  viewed as a generative problem that requires both semantic understanding over contextual conditions and stochastic modeling of continuous temporal dynamics. Existing approaches typically rely on either autoregressive large language models (LLMs) for semantic context modeling or diffusion-like models for continuous probabilistic generation. However, neither method alone can adequately model both aspects simultaneously. In this work, we propose CoGenCast, a hybrid generative framework that couples pre-trained LLMs with flow-matching mechanism for effective time series forecasting. Specifically, we reconfigure  pre-trained decoder-only LLMs into a  native forecasting encoder–decoder backbone by modifying only the attention topology, enabling bidirectional context encoding and causal representation generation. Building on this, a flow-matching mechanism  is further integrated to model temporal evolution, capturing continuous stochastic dynamics conditioned on the autoregressively generated representation. Notably, CoGenCast naturally supports multimodal forecasting and cross-domain unified training. Extensive experiments on multiple benchmarks show that CoGenCast consistently outperforms previous compared baselines. 
</details>

## 🖼️ Overview
![CoGenCast Framework](assets/framework.png)

## ✨ Key Features
- Coupled autoregressive-flow generation for expressive forecasting distributions
- Flexible backbone integration with large language models for sequence modeling
- Supports pretraining and finetuning across multiple time-series datasets

## 🚀 Quick Start
### Environment
- Python 3.10 (recommended)
```bash
conda create cogencast python=3.10
conda activate cogencast
```
- Install dependencies
```bash
pip install -r requirements.txt
pip install transformers
```

### Datasets and LLM Backbone
- Datasets: download the datasets you need and place them under `./datasets/` (e.g., `./datasets/ETTh1/ETTh1.csv`).
- Backbone: download the Qwen3-0.6B weights from [Hugging Face](https://huggingface.co/) and set the local path via `--llm_path`.

### Example Run
```bash
CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --task_name finetune \
  --is_training 1 \
  --root_path ./datasets/Wind/ \
  --data_path Wind.csv \
  --model_id Wind \
  --model CoGenCast \
  --data Wind \
  --features M \
  --input_len 96 \
  --label_len 0 \
  --pred_len 12 \
  --e_layers 2 \
  --pt_layers 4 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --n_heads 16 \
  --d_model 1024 \
  --d_ff 256 \
  --patch_len 4 \
  --stride 4 \
  --dropout 0.2 \
  --head_dropout 0.1 \
  --batch_size 4 \
  --gpu 0 \
  --lr_decay 0.5 \
  --lradj step \
  --time_steps 1000 \
  --scheduler cosine \
  --patience 3 \
  --backbone Qwen3-0.6B \
  --learning_rate 1e-4 \
  --pct_start 0.3
```

## 📊 Performance
![Main Results](assets/table.png)

## 🙏 Acknowledgement
This repo is built on the pioneer works. We appreciate the following GitHub repos a lot for their valuable code base or datasets:
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)  [LLM4TS](https://github.com/blacksnail789521/LLM4TS)
  [Time-LLM](https://github.com/KimMeen/Time-LLM)
  [FlowTS](https://github.com/UNITES-Lab/FlowTS)
  [CDPM](https://github.com/zjt-gpu/CDPM)
  [CSDI](https://github.com/ermongroup/CSDI)
  [TimeDART](https://github.com/Melmaphother/TimeDART)
  [PatchTST](https://github.com/yuqinie98/PatchTST)
