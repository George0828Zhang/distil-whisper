#!/usr/bin/env bash

python run_concatenate.py \
  --dataset_name "georgechang8/code_switch_yodas_zh" \
  --dataset_config_name "default" \
  --dataset_split_name "train" \
  --text_column_name "text" \
  --id_column_name "id" \
  --speaker_id_column_name "session_id" \
  --output_dir "./yodas_30s" \
  --preprocessing_batch_size 512 \
  --preprocessing_num_workers 6