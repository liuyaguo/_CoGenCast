for iscross in 0; do
    for ispretrain in 0; do
        for pred_len in 12; do
            for trial in 1; do
                # 遍历学习率
                for lr in  6e-5; do
                    CUDA_VISIBLE_DEVICES=1 python -u run.py \
                        --task_name finetune \
                        --is_training 1 \
                        --root_path ./datasets/Energy/ \
                        --data_path Energy.csv \
                        --model_id Energy \
                        --model CoGenCast \
                        --data Energy \
                        --features S \
                        --input_len 96\
                        --label_len 0 \
                        --pred_len $pred_len \
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
                        --batch_size 1 \
                        --gpu 0 \
                        --lr_decay 0.5 \
                        --lradj step \
                        --time_steps 1000 \
                        --scheduler cosine \
                        --patience 3 \
                        --backbone Qwen3-0.6B\
                        --learning_rate $lr  \
                        --pct_start 0.3 \
                        --trial $trial \
                        --ispretrain $ispretrain \
                        --iscross $iscross\
                        --text 1
                done
            done
        done
    done
done


