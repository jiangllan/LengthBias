

python ../../run_baseline.py \
  --model_name BiMPM \
  --data_dir ../dataset \
  --embedding_dir \
  --save_dir ../tmp/BiMPM \
  --analysis_mode rel \
  --train_batch_size 64 \
  --eval_batch_size 32 \
  --logging_step 1000 \
  --my_task qqp \
  --epoch_num 10 \
  --max_len 50 \
  --balanced_mode ratio_balanced \
  --delta_lens_mode rel_delta_lens \
  --do_train \
  --do_eval