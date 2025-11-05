import torch
import os
import yaml
from typing import Optional
from itertools import chain
from dataclasses import dataclass, field
from glob import glob
import argparse
from transformers import (
    TrainingArguments,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    Trainer
)
from transformers.integrations import is_deepspeed_zero3_enabled
from utils.data_loader import load_raw_datasets
from utils.tools import get_free_memory_per_device, get_model_parameters, print_trainable_parameters, find_all_linear_names



@dataclass
class DataArguments:
    train_data_path: Optional[str] = field(
        default="./code/Training-Box/data/pretrain", metadata={"help": "Path to the training data."}
    )
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=10,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    block_size: Optional[int] = field(
        default=100,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="./models/Qwen2.5-1.5B-Instruct",
        metadata={
            "help": "Model checkpoint for weights initialization"
        }
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path of tokenizer"
        }
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        }
    )
    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Options: ['auto', 'bfloat16', 'float16', 'float32']"
        }
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Device to map model to. If `auto` is passed, the device will be selected automatically. "},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    hf_hub_token: Optional[str] = field(default=None, metadata={"help": "Auth token to log in with Hugging Face Hub."})
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading a model from a remote checkpoint."},
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")


class PretrainBox:
    def __init__(self, data_args, model_args, train_args):
        self.data_args=data_args
        self.model_args=model_args
        self.train_args=train_args
        self.tokenizer = self.load_tokenizer()
        self.block_size = self.tokenizer.model_max_length if not data_args.block_size or self.tokenizer.model_max_length < data_args.block_size else data_args.block_size
        self.datasets=self.preprocess_data()

    def load_tokenizer(self):
        # Load tokenizer
        tokenizer_kwargs = {
            "cache_dir": self.model_args.cache_dir,
            "use_fast": self.model_args.use_fast_tokenizer,
            "trust_remote_code": self.model_args.trust_remote_code,
        }
        tokenizer_name_or_path = self.model_args.tokenizer_name_or_path
        if not tokenizer_name_or_path:
            tokenizer_name_or_path = self.model_args.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
        return tokenizer
    
    def load_model(self):
        # Load model
        config_kwargs = {
            "trust_remote_code": self.model_args.trust_remote_code,
            "cache_dir": self.model_args.cache_dir,
            "revision": self.model_args.model_revision,
            "token": self.model_args.hf_hub_token,
        }
        model_config = AutoConfig.from_pretrained(self.model_args.model_name_or_path, **config_kwargs)
        model_kwargs = {
            "low_cpu_mem_usage": not is_deepspeed_zero3_enabled(),
            "device_map": self.model_args.device_map,
            "torch_dtype": "auto"
        }

        model = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_name_or_path,
                config=model_config,   
                **model_kwargs
            )
        if '32' in self.model_args.torch_dtype:
            model_size_gib = get_model_parameters(model)*4/1024**3
        else:
            model_size_gib = get_model_parameters(model)*2/1024**3
        print(f"模型大小：{model_size_gib:.3f}GiB")
        if self.model_args.device_map != "auto":
            """DDP"""
            use_tensor_parallel = False
            print("单卡加载模型")
        elif torch.cuda.device_count() > 1 and sum(list(get_free_memory_per_device().values())) > model_size_gib:
            use_tensor_parallel = True
            print("多卡加载模型")
            model_kwargs["device_map"] = 'auto'
            model_kwargs["max_memory"] = {i: f"{v}GiB" for i, v in get_free_memory_per_device().items() if v > 10}
            print(f"使用设备显存空闲大小：{model_kwargs['max_memory']}")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_name_or_path,
                config=model_config,
                **model_kwargs
            )
            model.is_parallelizable = True
            model.model_parallel = True
        
        if self.train_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        else:
            model.config.use_cache = True
        # model.enable_input_require_grads()

        return model
    
    def tokenize_w_pad_function(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["text"],
            truncation=True,
            padding='max_length',
            max_length=self.block_size
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs

    def tokenize_wo_pad_function(self, examples):
        return self.tokenizer(examples["text"])

    def group_text_function(self, examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        result = {
            k: [t[i: i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def preprocess_data(self):
        raw_datasets = load_raw_datasets(self.data_args.train_data_path, self.data_args.eval_data_path, extension='jsonl', validation_split_percentage=self.data_args.validation_split_percentage, cache_dir=self.model_args.cache_dir)
        column_names = list(raw_datasets["train"].features)
        if not self.data_args.streaming:
            if self.train_args.group_by_length:
                tokenized_datasets = raw_datasets.map(
                    self.tokenize_wo_pad_function,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                )
                lm_datasets = tokenized_datasets.map(
                    self.group_text_function,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                )
            else:
                lm_datasets = raw_datasets.map(
                    self.tokenize_w_pad_function,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                )
        else:
            if train_args.group_by_length:
                tokenized_datasets = raw_datasets.map(
                    self.tokenize_wo_pad_function,
                    batched=True,
                    remove_columns=column_names,
                )
                lm_datasets = tokenized_datasets.map(
                    self.group_text_function,
                    batched=True,
                )
            else:
                lm_datasets = raw_datasets.map(
                    self.tokenize_w_pad_function,
                    batched=True,
                    remove_columns=column_names,
                )
        # print(lm_datasets)
        return lm_datasets.shuffle(self.train_args.seed)
        
    def train(self):
        model = self.load_model()
        print_trainable_parameters(model)
        trainer = Trainer(
            model=model,
            args=self.train_args,
            train_dataset=self.datasets["train"],
            eval_dataset=self.datasets["validation"],
        )
        trainer.train()

        

if __name__ == "__main__":
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    parser.add_argument("-f","--yaml-file", type=str, default="./yaml/continue_pretrain.yaml", help="The YAML configuration file for training")
    data_args, model_args, train_args, config_file = parser.parse_args_into_dataclasses()
    with open(config_file.yaml_file) as yf:
        config = yaml.safe_load(yf)
    if config:
        for k,v in config.items():
            if hasattr(data_args, k):
                setattr(data_args, k, v)
            elif hasattr(model_args, k):
                setattr(model_args, k, v)
            elif hasattr(train_args, k):
                setattr(train_args, k, v)
            else:
                print(f"Warning: The arg '{k}' in yaml config file is invalid!")
    print(data_args, '\n', model_args, '\n', train_args)
    set_seed(train_args.seed)
    pb = PretrainBox(data_args, model_args, train_args)
    pb.train()
    