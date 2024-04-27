#!/usr/bin/env bash

python run_concatenate.py \
  --dataset_name "georgechang8/ASCEND_CLEAN" \
  --dataset_split_name "train+validation+test" \
  --text_column_name "transcription" \
  --id_column_name "id" \
  --speaker_id_column_name "session_id" \
  --output_dir "./ascend_30s" \
  --preprocessing_batch_size 512 \
  --preprocessing_num_workers 6