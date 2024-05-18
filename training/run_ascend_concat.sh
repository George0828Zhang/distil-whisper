#!/usr/bin/env bash

python run_concatenate.py \
  --dataset_name "./ASCEND_CLEAN" \
  --dataset_config_name "default" \
  --dataset_split_name "test" \
  --text_column_name "transcription" \
  --id_column_name "id" \
  --speaker_id_column_name "session_id" \
  --output_dir "./ascend_test" \
  --preprocessing_batch_size 1024 \
  --preprocessing_num_workers 4 \
  --random_seed 1877 \
  --output_format "files" \
  --max_duration_in_seconds 9999999