#!/usr/bin/env bash

python run_concatenate.py \
  --dataset_name "mozilla-foundation/common_voice_16_1" \
  --dataset_config_name "zh-TW" \
  --dataset_split_name "train+validation" \
  --text_column_name "sentence" \
  --id_column_name "path" \
  --speaker_id_column_name "client_id" \
  --output_dir "./cv16_30s" \
  --preprocessing_batch_size 512 \
  --preprocessing_num_workers 6 \
  --randomize_same_speaker \
  --random_seed 1877