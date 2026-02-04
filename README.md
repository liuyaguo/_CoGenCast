<h1 align="center">CoGenCast: A Coupled Autoregressive-Flow Generative Framework<br>for Time Series Forecasting</h1>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.2.2" />
  <img src="https://img.shields.io/badge/Time%20Series-Forecasting-2F4F4F" alt="Time Series Forecasting" />
</p>

## 📝 Abstract
We propose CoGenCast, a coupled autoregressive-flow generative framework for time series forecasting. The method integrates an autoregressive backbone with flow-based generation to model complex temporal dependencies, aiming to improve forecasting accuracy and robustness across diverse datasets.

## 🖼️ Overview
![CoGenCast Framework](assets/framework.png)

## ✨ Key Features
- Coupled autoregressive-flow generation for expressive forecasting distributions
- Flexible backbone integration with large language models for sequence modeling
- Supports pretraining and finetuning across multiple time-series datasets

## 🚀 Quick Start
### Environment
- Python 3.10 (recommended)
- Install dependencies:

```bash
pip install -r requirements.txt
pip install transformers
```

### Datasets and Qwen3-0.6B
- Datasets: download the datasets you need and place them under `./datasets/` (e.g., `./datasets/ETTh1/ETTh1.csv`).
- Qwen3-0.6B: download the Qwen3-0.6B weights from [Hugging Face](https://huggingface.co/) and set the local path via `--llm_path`.

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
-Time-Series-Library
