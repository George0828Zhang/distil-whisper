#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pseudo-labelling audio data using the Whisper model in preparation for distillation.
"""
# You can also adapt this script for your own pseudo-labelling tasks. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
import transformers
from datasets import (
    DatasetDict,
    load_dataset,
)
from huggingface_hub import HfFolder
from transformers import HfArgumentParser
from transformers.utils.versions import require_version


require_version("datasets>=2.14.6", "To fix: `pip install --upgrade datasets`")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to cache directory for saving and loading datasets"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    preprocessing_batch_size: Optional[int] = field(
        default=500,
        metadata={"help": "The batch size to use for the dataset pre-processing."},
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'."},
    )
    id_column_name: str = field(
        default="id",
        metadata={"help": "The name of the dataset column containing the id data. Defaults to 'id'"},
    )
    speaker_id_column_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset column containing the speaker id data. Defaults to None."},
    )
    max_duration_in_seconds: float = field(
        default=30.0,
        metadata={"help": "Filter audio files that are longer than `max_duration_in_seconds` seconds"},
    )
    sampling_rate: int = field(
        default=16000,
        metadata={"help": "Sampling rate of output dataset"},
    )
    dataset_split_name: str = field(
        default="train+validation+test",
        metadata={
            "help": (
                "The name of the data set splits to use (via the datasets library)."
                " Defaults to 'train+validation+test'. Multiple splits can be passed by splitting a"
                " list through the '+' character, e.g. 'train+validation' will"
                " pseudo-label both the 'train' and 'validation' splits sequentially."
            )
        },
    )
    max_samples_per_split: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples per split to this value if set."},
    )
    output_dir: str = field(
        default="./output_30s",
        metadata={
            "help": "output dir."
        },
    )


def main():
    # 1. Parse input arguments
    # We keep distinct sets of args, for cleaner separation of model/data/training related args
    parser = HfArgumentParser((DataTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args, = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args, = parser.parse_args_into_dataclasses()

    # 3. Set-up basic logging
    # Create one log on every process with the configuration for debugging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Set the verbosity to info of the Transformers logger (on main process only)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    # 3. Load dataset
    raw_datasets = DatasetDict()
    token = data_args.token if data_args.token is not None else HfFolder().get_token()

    data_splits = data_args.dataset_split_name.split("+")
    for split in data_splits:
        raw_datasets[split] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=split,
            cache_dir=data_args.dataset_cache_dir,
            token=token,
            streaming=False,
            num_proc=data_args.preprocessing_num_workers,
        )

    if data_args.audio_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--audio_column_name '{data_args.audio_column_name}' not found in dataset"
            f" '{data_args.dataset_name}'. Make sure to set `--audio_column_name` to"
            " the correct audio column - one of"
            f" {', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    if data_args.text_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--text_column_name {data_args.text_column_name} not found in dataset"
            f" '{data_args.dataset_name}'. Make sure to set `--text_column_name` to the"
            " correct text column - one of"
            f" {', '.join(next(iter(raw_datasets.values())).column_names)}."
        )
    
    sampling_rate = data_args.sampling_rate

    # 6. Resample speech dataset: `datasets` takes care of automatically loading and resampling the audio,
    # so we just need to set the correct target sampling rate.
    raw_datasets = raw_datasets.cast_column(
        data_args.audio_column_name,
        datasets.features.Audio(sampling_rate=sampling_rate),
    )

    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = int(data_args.max_duration_in_seconds * sampling_rate)
    audio_column_name = data_args.audio_column_name

    preprocessing_batch_size = data_args.preprocessing_batch_size
    num_workers = data_args.preprocessing_num_workers

    text_column_name = data_args.text_column_name
    id_column_name = data_args.id_column_name
    speaker_id_column_name = data_args.speaker_id_column_name

    if data_args.max_samples_per_split is not None:
        for split in data_splits:
            raw_datasets[split] = raw_datasets[split].select(range(data_args.max_samples_per_split))

    if speaker_id_column_name is not None:
        raw_datasets = raw_datasets.sort([speaker_id_column_name, id_column_name])

    def concatenate_dataset(batch):
        audio = [sample["array"] for sample in batch[audio_column_name]]
        input_lengths = [len(sample) for sample in audio]

        text = batch[text_column_name]
        speaker_id = batch[speaker_id_column_name] if speaker_id_column_name else len(text) * [None]

        concatenated_audio = []
        concatenated_text = []
        concatenated_speaker = []

        def get_timestamp(s, p=0.02):
            return "<|%.2f|>" % (int(s / p)*p)
        
        def get_timestamped_text(t, start, end):
            return f"{get_timestamp(start)}{t}{get_timestamp(end)}"
        
        def get_start(audio):
            from stable_whisper.stabilization.nonvad import audio2timings
            out = audio2timings(
                audio[:sampling_rate // 2],  # first 0.5 second
                sr=sampling_rate
            )
            if out is None or out[0][0] > 0.02:
                return 0
            return out[1][0]

        def concat_with_silence(utt1, utt2, sil):
            sil = min(int(sil/2), len(utt1), len(utt2))
            true_sil = np.zeros(sil, dtype=utt1.dtype)

            weight = np.linspace(1, 0, sil)
            silence1 = utt1[-sil:] * weight + true_sil * (1 - weight)
            silence2 = true_sil * weight + utt2[:sil] * (1 - weight)

            out = np.concatenate([utt1, silence1, silence2, utt2])
            return out


        audio_sample = audio[0]
        end_ts = len(audio_sample) / sampling_rate
        text_sample = get_timestamped_text(text[0], get_start(audio_sample), end_ts)

        for idx in range(1, len(audio)):
            prev_speaker = speaker_id[idx - 1]
            speaker = speaker_id[idx]
            sil_samples = np.random.rand() * sampling_rate

            if len(audio_sample) + input_lengths[idx] + sil_samples <= max_input_length:
                if speaker == prev_speaker:
                    # we have no information about whether the segments follow on sequentially
                    # so we just ensure the same speaker as we concatenate across files
                    audio_sample = concat_with_silence(audio_sample, audio[idx], sil_samples)
                    end_ts = len(audio_sample) / sampling_rate
                    # extra spaces in the text transcription don't matter, since we only use it for the WER computation
                    text_sample += get_timestamped_text(text[idx], end_ts - input_lengths[idx]/sampling_rate, end_ts)
                else:
                    # speakers do not follow sequentially, save the audio and start looping again
                    concatenated_audio.append(audio_sample)
                    concatenated_text.append(text_sample)
                    concatenated_speaker.append(speaker)
                    audio_sample = audio[idx]
                    end_ts = len(audio_sample) / sampling_rate
                    text_sample = get_timestamped_text(text[idx], get_start(audio_sample), end_ts)

            else:
                # concatenated audio exceeds max length, save the audio and start looping again
                concatenated_audio.append(audio_sample)
                concatenated_text.append(text_sample)
                concatenated_speaker.append(speaker)
                audio_sample = audio[idx]
                end_ts = len(audio_sample) / sampling_rate
                text_sample = get_timestamped_text(text[idx], get_start(audio_sample), end_ts)

        batch[audio_column_name] = [{"array": array, "sampling_rate": sampling_rate} for array in concatenated_audio]
        batch[text_column_name] = concatenated_text
        batch[id_column_name] = concatenated_speaker
        batch["condition_on_prev"] = [False] + [
            a == b
            for a, b in zip(concatenated_speaker[:-1], concatenated_speaker[1:])
        ]
        return batch

    raw_datasets_features = list(next(iter(raw_datasets.values())).features.keys())

    raw_datasets = raw_datasets.map(
        concatenate_dataset,
        batched=True,
        batch_size=preprocessing_batch_size,
        num_proc=num_workers,
        remove_columns=set(raw_datasets_features)
        - {audio_column_name, text_column_name, id_column_name, "condition_on_prev"},
        desc="Concatenating dataset...",
    )

    raw_datasets = raw_datasets.cast_column(
        audio_column_name, datasets.features.Audio(sampling_rate=sampling_rate)
    )
    pretty_name = data_args.dataset_name.split("/")[-1]

    def postprocess_ids(speaker_ids, indices):
        speaker_ids_formatted = []
        for speaker, idx in zip(speaker_ids, indices):
            formatted_idx = f"{pretty_name}-{speaker}-{idx}" if speaker is not None else f"{pretty_name}-{idx}"
            speaker_ids_formatted.append(formatted_idx)
        return {id_column_name: speaker_ids_formatted}

    raw_datasets = raw_datasets.map(
        postprocess_ids,
        input_columns=[id_column_name],
        with_indices=True,
        desc="Setting sample idxs...",
        batched=True,
        batch_size=preprocessing_batch_size,
        num_proc=num_workers,
    )

    output_dir = data_args.output_dir
    # this is where we'll save our transcriptions
    os.makedirs(output_dir, exist_ok=True)

    raw_datasets.save_to_disk(output_dir, num_proc=num_workers)

if __name__ == "__main__":
    main()
