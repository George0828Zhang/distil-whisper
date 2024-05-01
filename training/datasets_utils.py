from datasets import DatasetDict
from datasets import config
from datasets.utils.py_utils import convert_file_size_to_int
from datasets.table import embed_table_storage
from tqdm import tqdm
from pathlib import Path
import yaml
import re

def load_readme(path):
    with open(path, "r") as f:
        yaml_data = re.search(r"---((.|\n)+)---", f.read()).group(1)
        data = yaml.safe_load(yaml_data)
    return data

def dump_readme(data, path):
    with open(path, "w") as f:
        yaml_data = f"---\n{yaml.dump(data).strip()}\n---"
        f.write(yaml_data)

def save_to_disk_as_parquet(dataset, data_dir, config_name="default", split='train', max_shard_size='1GB'):
    """source: https://discuss.huggingface.co/t/how-to-save-audio-dataset-with-parquet-format-on-disk/66179
    Saves data to disk as parquet, which can be loaded by load_dataset
    """
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
    if isinstance(dataset, DatasetDict):
        config_dir = "data" if config_name == "default" else config_name
        for split in dataset:
            save_to_disk_as_parquet(dataset[split], data_dir / config_dir, split=split, max_shard_size=max_shard_size)
        try:
            data = load_readme(data_dir / "README.md")
        except (FileNotFoundError, AttributeError):
            data = dict(configs=[])

        data.get('configs', []).append(dict(
            config_name=config_name,
            data_files=[
                dict(split=split, path=f'{config_dir}/{split}-*')
                for split in dataset
            ]
        ))
        dump_readme(data, data_dir / "README.md")
        return

    dataset_nbytes = dataset._estimate_nbytes()
    max_shard_size = convert_file_size_to_int(max_shard_size or config.MAX_SHARD_SIZE)
    num_shards = int(dataset_nbytes / max_shard_size) + 1
    num_shards = max(num_shards, 1)
    shards = (dataset.shard(num_shards=num_shards, index=i, contiguous=True) for i in range(num_shards))

    def shards_with_embedded_external_files(shards):
        for shard in shards:
            format = shard.format
            shard = shard.with_format("arrow")
            shard = shard.map(
                embed_table_storage,
                batched=True,
                batch_size=1000,
                keep_in_memory=True,
            )
            shard = shard.with_format(**format)
            yield shard
    shards = shards_with_embedded_external_files(shards)

    data_dir.mkdir(parents=True, exist_ok=True)

    for index, shard in tqdm(
        enumerate(shards),
        desc="Save the dataset shards",
        total=num_shards,
    ):
        shard_path = data_dir / f"{split}-{index:05d}-of-{num_shards:05d}.parquet"
        shard.to_parquet(str(shard_path))