import torch
import os
import yaml
import copy
from typing import Optional
from itertools import chain
from dataclasses import dataclass, field
from glob import glob
import argparse
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    AutoProcessor,
    HfArgumentParser,
    BitsAndBytesConfig,
    set_seed,
    Trainer,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info
from qwen_omni_utils import process_mm_info, process_vision_info
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, PeftModel, get_peft_model
from trl import SFTTrainer, SFTConfig
from transformers.integrations import is_deepspeed_zero3_enabled
from utils.data_loader import load_raw_datasets
from utils.tools import get_free_memory_per_device, get_model_parameters, print_trainable_parameters, find_all_linear_names
from utils.formats import convert_to_standard_openai_format, is_openai_format



@dataclass
class DataArguments:
    train_data_path: Optional[str] = field(
        default="./code/Training-Box/data/rl/dpo", metadata={"help": "Path to the training data."}
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

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="./models/Qwen2.5-VL-3B-Instruct",
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
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the model in 8bit mode or not."})
    load_in_4bit: bool = field(default=False, metadata={"help": "Whether to load the model in 4bit mode or not."})
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

@dataclass
class ScriptArguments:
    use_peft: bool = field(default=True, metadata={"help": "Whether to use peft"})
    target_modules: Optional[str] = field(default="all", metadata={"help": "Options: ['q_proj', 'k_proj', 'v_proj', 'o_proj'], Specific details vary depending on the model."})
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None)


class SFTVBox:
    def __init__(self, data_args, model_args, train_args, lora_args):
        self.data_args=data_args
        self.model_args=model_args
        self.train_args=train_args
        self.lora_args=lora_args
        self.processor = self.load_processor()
        self.tokenizer = self.load_tokenizer()
        self.datasets=self.preprocess_data()

    def load_processor(self):
        # Load processor for multi-modal
        tokenizer_kwargs = {
            "cache_dir": self.model_args.cache_dir,
            "use_fast": self.model_args.use_fast_tokenizer,
            "trust_remote_code": self.model_args.trust_remote_code,
        }
        tokenizer_name_or_path = self.model_args.tokenizer_name_or_path
        if not tokenizer_name_or_path:
            tokenizer_name_or_path = self.model_args.model_name_or_path
        tokenizer = AutoProcessor.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
        return tokenizer

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
        model_config = Qwen2_5_VLConfig.from_pretrained(self.model_args.model_name_or_path, **config_kwargs)
        model_kwargs = {
            "low_cpu_mem_usage": not is_deepspeed_zero3_enabled(),
            "device_map": self.model_args.device_map,
            "torch_dtype": "auto"
        }
        if self.model_args.load_in_8bit or self.model_args.load_in_4bit:
            if self.model_args.load_in_8bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=self.model_args.load_in_8bit
                )
            elif self.model_args.load_in_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=self.model_args.load_in_4bit,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.model_args.torch_dtype,
                )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_args.model_name_or_path,
                config=model_config,
                **model_kwargs
            )
            model.is_parallelizable = True
            model.model_parallel = True
        
        if train_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        else:
            model.config.use_cache = True
        # model.enable_input_require_grads()

        return model
    
    def preprocess_func(self, examples):
        new_examples = {
            "input_ids":[],
            "attention_mask":[],
            "pixel_values": [],
            "image_grid_thw": [],
            "labels":[]
        }
        messages = examples["messages"]
        images = examples.get("images")
        videos = examples.get("videos")

        if images and videos:
            for idx, (m, i, v) in enumerate(zip(messages, images, videos)):
                sample = {
                    "messages": m,
                    "images": i,
                    "videos": v
                }
                new_sample = convert_to_standard_openai_format(sample)
                text = self.processor.apply_chat_template(new_sample["messages"][:-1], add_generation_prompt=True, tokenize=False, add_vision_id=True)
                response = new_sample["messages"][-1]["content"]
                res = self.tokenizer(response, add_special_tokens=False)
                image_pix, video_pix = process_vision_info(new_sample["messages"][:-1])
                inputs = self.processor(
                    text = [text],
                    images=image_pix,
                    videos=video_pix,
                    padding=True
                )
                input_ids = inputs["input_ids"][0]+res["input_ids"]+[self.tokenizer.eos_token_id]
                attention_mask = inputs["attention_mask"][0]+res["attention_mask"]+[1]
                labels = [LabelSmoother.ignore_index]*len(inputs["input_ids"][0])+res["input_ids"]+[self.tokenizer.eos_token_id]
                new_examples["input_ids"].append(torch.tensor(input_ids))
                new_examples["attention_mask"].append(torch.tensor(attention_mask))
                new_examples["labels"].append(torch.tensor(labels))
                new_examples["pixel_values"].append(torch.tensor(inputs["pixel_values"]))
                new_examples["image_grid_thw"].append(torch.tensor(inputs["image_grid_thw"]))
            return new_examples
        elif images:
            for idx, (m, i) in enumerate(zip(messages, images)):
                sample = {
                    "messages": m,
                    "images": i
                }
                new_sample = convert_to_standard_openai_format(sample, resize_image=None)
                text = self.processor.apply_chat_template(new_sample["messages"][:-1], add_generation_prompt=True, tokenize=False, add_vision_id=True)
                response = new_sample["messages"][-1]["content"]
                res = self.tokenizer(response, add_special_tokens=False)
                image_pix, _ = process_vision_info(new_sample["messages"][:-1])        
                inputs = self.processor(
                    text = [text],
                    images=image_pix,
                    videos=None,
                    padding=True
                )

                input_ids = inputs["input_ids"][0]+res["input_ids"]+[self.tokenizer.eos_token_id]
                attention_mask = inputs["attention_mask"][0]+res["attention_mask"]+[1]
                labels = [LabelSmoother.ignore_index]*len(inputs["input_ids"][0])+res["input_ids"]+[self.tokenizer.eos_token_id]
                new_examples["input_ids"].append(torch.tensor(input_ids))
                new_examples["attention_mask"].append(torch.tensor(attention_mask))
                new_examples["labels"].append(torch.tensor(labels))
                new_examples["pixel_values"].append(torch.tensor(inputs["pixel_values"]))
                new_examples["image_grid_thw"].append(torch.tensor(inputs["image_grid_thw"]))
            return new_examples
        elif videos:
            for idx, (m, v) in enumerate(zip(messages, videos)):
                sample = {
                    "messages": m,
                    "videos": i
                }
                new_sample = convert_to_standard_openai_format(sample)
                text = self.processor.apply_chat_template(new_sample["messages"][:-1], add_generation_prompt=True, tokenize=False, add_vision_id=True)
                response = new_sample["messages"][-1]["content"]
                res = self.tokenizer(response, add_special_tokens=False)
                _, video_pix = process_vision_info(new_sample["messages"][:-1])
                inputs = self.processor(
                    text = [text],
                    images=None,
                    videos=video_pix,
                    padding=True
                )
                input_ids = inputs["input_ids"][0]+res["input_ids"]+[self.tokenizer.eos_token_id]
                attention_mask = inputs["attention_mask"][0]+res["attention_mask"]+[1]
                labels = [LabelSmoother.ignore_index]*len(inputs["input_ids"][0])+res["input_ids"]+[self.tokenizer.eos_token_id]
                new_examples["input_ids"].append(torch.tensor(input_ids))
                new_examples["attention_mask"].append(torch.tensor(attention_mask))
                new_examples["labels"].append(torch.tensor(labels))
                new_examples["pixel_values"].append(torch.tensor(inputs["pixel_values"]))
                new_examples["image_grid_thw"].append(torch.tensor(inputs["image_grid_thw"]))
            return new_examples
        else:
            for idx, m in enumerate(messages):
                sample = {
                    "messages": m,
                }
                new_sample = convert_to_standard_openai_format(sample)
                text = self.processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                response = new_sample["messages"][-1]["content"]
                res = self.tokenizer(response, add_special_tokens=False)
                inputs = self.processor(
                    text = [text],
                    images=None,
                    videos=None,
                    padding=True
                )
                input_ids = inputs["input_ids"][0]+res["input_ids"]+[self.tokenizer.eos_token_id]
                attention_mask = inputs["attention_mask"][0]+res["attention_mask"]+[1]
                labels = [LabelSmoother.ignore_index]*len(inputs["input_ids"][0])+res["input_ids"]+[self.tokenizer.eos_token_id]
                new_examples["input_ids"].append(torch.tensor(input_ids))
                new_examples["attention_mask"].append(torch.tensor(attention_mask))
                new_examples["labels"].append(torch.tensor(labels))
                new_examples["pixel_values"].append(torch.tensor(inputs["pixel_values"]))
                new_examples["image_grid_thw"].append(torch.tensor(inputs["image_grid_thw"]))
            return new_examples

    def preprocess_data(self):
        raw_datasets = load_raw_datasets(self.data_args.train_data_path, self.data_args.eval_data_path, extension='json', validation_split_percentage=self.data_args.validation_split_percentage, cache_dir=self.model_args.cache_dir)
        column_names = list(raw_datasets["train"].features)
        if not self.data_args.streaming:
            datasets = raw_datasets.shuffle(self.train_args.seed).map(
                self.preprocess_func,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
            )
        else:
            datasets = raw_datasets.shuffle(self.train_args.seed).map(
                self.preprocess_func,
                batched=True,
                remove_columns=column_names,
            )

        return datasets

    def train(self):
        model = self.load_model()
        if self.lora_args.use_peft:
            if self.model_args.load_in_8bit or self.model_args.load_in_4bit:
                print("Use QLoRA(PEFT)")
                model = prepare_model_for_kbit_training(model, self.train_args.gradient_checkpointing)
            else:
                print("Use LoRA(PEFT)")
            if self.lora_args.peft_path is not None:
                model = PeftModel.from_pretrained(model, self.lora_args.peft_path, is_trainable=True)
            else:
                target_modules = self.lora_args.target_modules.split(',') if self.lora_args.target_modules else None
                if target_modules and 'all' in target_modules:
                    target_modules = find_all_linear_names(model, int4=self.model_args.load_in_4bit, int8=self.model_args.load_in_8bit)
                print(f"Peft target_modules: {target_modules}")
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=target_modules,
                    inference_mode=False,
                    r=self.lora_args.lora_rank,
                    lora_alpha=self.lora_args.lora_alpha,
                    lora_dropout=self.lora_args.lora_dropout,
                )
                model = get_peft_model(model, peft_config)

        trainer = SFTTrainer(
            model=model,
            args=self.train_args,
            train_dataset=self.datasets["train"],
            eval_dataset=self.datasets["validation"],
            processing_class=self.tokenizer,
        )
        print_trainable_parameters(model)
        trainer.train()
       

if __name__ == "__main__":
    parser = HfArgumentParser((DataArguments, ModelArguments, SFTConfig, ScriptArguments))
    parser.add_argument("-f","--yaml-file", type=str, default="./yaml/sft_v.yaml", help="The YAML configuration file for training")
    data_args, model_args, train_args, lora_args, config_file = parser.parse_args_into_dataclasses()
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
    print(data_args, '\n', model_args, '\n', train_args, '\n', lora_args)
    set_seed(train_args.seed)
    sftv_b = SFTVBox(data_args, model_args, train_args, lora_args)
    sftv_b.train()