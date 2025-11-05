from datasets import load_dataset
from glob import glob
import os
import pandas as pd
import json
from typing import Optional,Literal

def load_raw_datasets(train_data_path: str, eval_data_path: Optional[str]=None, extension: Literal['txt', 'json', 'jsonl']='json', validation_split_percentage: int=10, cache_dir: Optional[str]=None):
        if train_data_path is None:
            raise ValueError("train_data_path must be exist")
        if extension == 'jsonl'or extension == 'json':
            ext = 'json'
        elif extension == 'txt':
            ext = 'text'
        else:
            raise ValueError(f"File extension {extension.lower()} is not supported!")
        data_files = {}
        if train_data_path is not None and os.path.exists(train_data_path):
            if os.path.isfile(train_data_path):
                 if os.path.splitext(train_data_path)[1].lower() != f'.{extension.lower()}':
                    raise ValueError(f"File extension must be {extension.lower()}")
                 train_data_files = [train_data_path]
            else:
                train_data_files = glob(f'{train_data_path}/**/*.{extension.lower()}', recursive=True)
            if len(train_data_files) == 0:
                raise ValueError("jsonl_file is not exist in train_data_path!")
            data_files["train"] = train_data_files
        if eval_data_path is not None and os.path.exists(eval_data_path):
            if os.path.isfile(eval_data_path):
                 if os.path.splitext(eval_data_path)[1].lower() != f'.{extension.lower()}':
                    raise ValueError(f"File extension must be {extension.lower()}")
                 eval_data_files = [eval_data_path]
            else:
                eval_data_files = glob(f'{eval_data_path}/**/*.{extension.lower()}', recursive=True)
            if len(eval_data_files) == 0:
                raise ValueError("jsonl_file is not exist in eval_data_path!")
            data_files["validation"] = eval_data_files
        
        raw_datasets = load_dataset(
            ext,
            data_files=data_files,
            cache_dir=cache_dir
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                ext,
                data_files=data_files,
                split=f"train[:{validation_split_percentage}%]",
                cache_dir=cache_dir,
            )

            # raw_datasets["train"] = load_dataset(
            #     ext,
            #     data_files=data_files,
            #     split=f"train[{validation_split_percentage}%:]",
            #     cache_dir=cache_dir,
            # )

        # print(raw_datasets)
        return raw_datasets

def load_parquet_data(parquet_file_path: str, is_save: bool=False):
    data = pd.read_parquet(parquet_file_path)
    json_data = data.to_json(orient='records', force_ascii=False)
    if is_save:
        json_file_path = os.path.splitext(parquet_file_path)[0]+'.json'
        data.to_json(json_file_path, orient='records', force_ascii=False, indent=2)
    return json.loads(json_data)
