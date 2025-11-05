import torch
import os
import psutil

def get_used_memory_per_device():
    data_info = {}
    num_gpu = torch.cuda.device_count()
    for i in range(num_gpu):
        byte_mem = torch.cuda.device_memory_used(i)
        data_info[i] = byte_mem/(1024**3)
    # data_info["cpu"] = psutil.virtual_memory().used/1024**3
    # print(data_info)
    return data_info

def get_free_memory_per_device():
    data_info = {}
    num_gpu = torch.cuda.device_count()
    for i in range(num_gpu):
        total_mem = torch.cuda.get_device_properties(i).total_memory/1024**3
        used_mem = torch.cuda.device_memory_used(i)/1024**3
        data_info[i] = total_mem-used_mem
    # data_info["cpu"] = psutil.virtual_memory().free/1024**3
    # print(data_info)
    return data_info

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            if 'visual' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)

def get_model_parameters(model):
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
    return all_param


def compute_metrics_for_nlp(eval_preds):
    pass

def compute_metrics_for_cls(eval_preds):
    pass

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

# get_used_memory_per_device()
# get_free_memory_per_device()