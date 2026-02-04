for lr in    1e-4 ; do
    CUDA_VISIBLE_DEVICES=4 \
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./datasets/exchange_rate/ \
        --data_path exchange_rate.csv \
        --model_id Exchange \
        --model CoGenCast \
        --data Exchange \
        --features M \
        --input_len 96 \
        --label_len 0 \
        --pred_len 12 \
        --e_layers 2 \
        --pt_layers 4 \
        --enc_in 8 \
        --dec_in 8 \
        --c_out 8 \
        --n_heads 16 \
        --d_model 1024 \
        --d_ff 256 \
        --patch_len 3 \
        --stride 3 \
        --dropout 0.2 \
        --head_dropout 0.1 \
        --batch_size 2 \
        --gpu 0 \
        --lr_decay 0.5 \
        --lradj step \
        --time_steps 1000 \
        --scheduler cosine \
        --patience 3 \
        --learning_rate $lr \
        --backbone Qwen3-0.6B \
        --pct_start 0.3 \
        --iscross 0 \
        --ispretrain 0
done