from typing import Dict, List, Tuple, Literal, Optional
import re
from copy import deepcopy

def is_multimodal_format(sample: Dict):
    if ('images' in sample and isinstance(sample['images'], List)) or ('videos' in sample and isinstance(sample['videos'], List)) or ('audios' in sample and isinstance(sample['audios'], List)):
        return True
    else:
        return False

def is_toolcall_format(sample: Dict):
    if 'tools' in sample:
        return True
    else:
        return False

def is_openai_format(sample: Dict):
    is_toolcall = is_toolcall_format(sample)
    if is_toolcall:
        roles = ["system", "user", "assistant" , "tool"]
    else:
        roles = ["system", "user", "assistant"]
    if "messages" in sample and isinstance(sample["messages"], List) and len(sample["messages"]) > 0:
        before_role = ""
        for idx, dialog in enumerate(sample["messages"]):
            if "role" not in dialog or "content" not in dialog:
                return False
            if idx != 0 and dialog.get("role") == "system":
                return False
            if dialog.get("role") not in roles:
                return False
            if idx != 0 and dialog.get("role") == "assistant" and before_role not in ("user", "tool"):
                return False
            elif idx != 0 and sample["messages"][0].get("role") != "system" and dialog.get("role") in ("user", "tool") and before_role != "assistant":
                return False
            before_role = dialog.get("role")
            if is_toolcall and dialog["role"] == "assistant" and "tool_calls" in dialog:
                if isinstance(dialog.get("tool_calls"), List) and len(dialog.get("tool_calls")) > 0:
                    judge_func = [not (i.get("function") and i["function"].get("name") and i["function"].get("arguments")) for i in dialog.get("tool_calls")]
                    if any(judge_func):
                        return False
                else:
                    return False
        return True
    else:
        return False


def is_sharegpt_format(sample: Dict):
    if is_toolcall_format(sample):
        roles = ["system", "human", "gpt" , "function_call", "observation"]
    else:
        roles = ["system", "human", "gpt"]
    if "conversations" in sample and isinstance(sample["conversations"], List) and len(sample["conversations"]) > 0:
        before_role = ""
        for idx, dialog in enumerate(sample["conversations"]):
            if "from" not in dialog or "value" not in dialog:
                return False
            if idx != 0 and dialog.get("from") == "system":
                return False
            if dialog.get("from") not in roles:
                return False
            if idx != 0 and dialog.get("from") == "gpt" and before_role not in ("human", "observation"):
                return False
            elif idx != 0 and dialog.get("from") == "function_call" and before_role != "human":
                return False
            elif idx != 0 and dialog.get("from") == "observation" and before_role != "function_call":
                return False
            elif idx != 0 and sample["conversations"][0].get("from") != "system" and dialog.get("from") == "human" and before_role != "gpt":
                return False
            before_role = dialog.get("from")
        return True
    else:
        return False


def is_alpaca_format(sample: Dict, task: Literal['SFT', 'DPO', 'KTO']='SFT'):
    if task.upper() == 'SFT':
        if 'instruction' in sample and 'input' in sample and 'output' in sample and isinstance(sample['instruction'], str) and isinstance(sample['input'], str) and isinstance(sample['output'], str):
            if not sample.get("instruction") and not sample.get("input"):
                return False
            if 'system' in sample and not isinstance(sample['instruction'], str):
                return False
            if 'history' in sample and not isinstance(sample['history'], List):
                return False
            elif 'history' in sample and isinstance(sample['history'], List):
                if len(sample['history']) > 0:
                    for dialog in sample['history']:
                        if not isinstance(dialog, (List, Tuple)) or len(dialog) != 2:
                            return False
            return True
        else:
            return False
    elif task.upper() == 'DPO':
        if 'instruction' in sample and 'input' in sample and 'chosen' in sample and 'rejected' in sample and isinstance(sample['instruction'], str) and isinstance(sample['input'], str) and isinstance(sample['chosen'], str) and isinstance(sample['rejected'], str):
            if not sample.get("instruction") and not sample.get("input"):
                return False
            if 'system' in sample and not isinstance(sample['instruction'], str):
                return False
            if 'history' in sample and not isinstance(sample['history'], List):
                return False
            elif 'history' in sample and isinstance(sample['history'], List):
                if len(sample['history']) > 0:
                    for dialog in sample['history']:
                        if not isinstance(dialog, (List, Tuple)) or len(dialog) != 2:
                            return False
            return True
        else:
            return False
    elif task.upper() == 'KTO':
        if 'instruction' in sample and 'input' in sample and 'output' in sample and 'label' in sample and isinstance(sample['instruction'], str) and isinstance(sample['input'], str) and isinstance(sample['output'], str) and isinstance(sample['label'], bool):
            if not sample.get("instruction") and not sample.get("input"):
                return False
            if 'system' in sample and not isinstance(sample['instruction'], str):
                return False
            if 'history' in sample and not isinstance(sample['history'], List):
                return False
            elif 'history' in sample and isinstance(sample['history'], List):
                if len(sample['history']) > 0:
                    for dialog in sample['history']:
                        if not isinstance(dialog, (List, Tuple)) or len(dialog) != 2:
                            return False
            return True
        else:
            return False
    else:
        raise ValueError(f"Task options ['SFT', 'DPO', 'KTO'] are supported currently.")
    
def format_alpaca_to_openai(sample: Dict):
    if not is_alpaca_format(sample, "SFT") and not is_alpaca_format(sample, "DPO") and not is_alpaca_format(sample, "KTO"):
        raise ValueError('The input format is not alpaca!')
    has_system_msg = True if sample.get("system") else False
    has_history_msg = True if sample.get("history") and len(sample.get("history")) > 0 else False
    new_sample = deepcopy(sample)
    query = sample.get("instruction")+sample.get("input")
    messages = []
    if has_system_msg:
        messages.append(
            {"role": "system", "content": sample.get("system")}
        )
    if "system" in new_sample: 
        del new_sample["system"]
    if has_history_msg:
        for dialog in sample.get("history"):
            messages.append(
                {"role": "user", "content": dialog[0]},
                {"role": "content", "content": dialog[1]},
            )
    if "history" in new_sample: 
        del new_sample["history"]
    messages.append(
        {"role": "user", "content": query}
    )
    if is_alpaca_format(sample, "SFT") or is_alpaca_format(sample, "KTO"):
        messages.append(
            {"role": "assistant", "content": sample.get("output")}
        )
        del new_sample["output"]
    else:
        messages.append(
            {"role": "assistant", "content": sample.get("chosen")}
        )
    del new_sample["instruction"], new_sample["input"]
    new_sample["messages"] = messages
    return new_sample

def split_sentence_by_tags(sentence: str, tags: List[str]=['<image>', '<video>', '<audio>']) -> List[str]:
    pattern = '|'.join(tags)
    parts = re.split(rf'({pattern})', sentence)
    results = [part for part in parts if part]
    return results

def convert_to_standard_openai_format(sample: Dict, map_multimodal_tag: Dict[Literal['image', 'video', 'audio'], str]={'image': '<image>', 'video': '<video>', 'audio': '<audio>'}, resize_image: Optional[Tuple[int, int]]=None, resize_video: Optional[Tuple[int, int]]=None, fps_video: Optional[float]=None):
    if is_alpaca_format(sample, "SFT") or is_alpaca_format(sample, "DPO") or is_alpaca_format(sample, "KTO"):
        sample = format_alpaca_to_openai(sample)
    is_toolcall = is_toolcall_format(sample)
    if not is_multimodal_format(sample):
        if is_openai_format(sample):
            judge_system = [i["role"] == "system" for i in sample["messages"][1:]]
            if any(judge_system):
                if sample["messages"][0]["role"] == "system":
                    raise ValueError('The system message must be only one!')
                else:
                    raise ValueError('The system message must be first message in conversations!')
            else:
                return sample
        elif is_sharegpt_format(sample):
            if sample["conversations"][0]["from"]=='system':
                messages = sample["conversations"][1:]
                new_sample = {
                    "messages": [{"role": "system", "content": sample["conversations"][0]["value"]}]
                }
            else:
                messages = sample["conversations"]
                new_sample = {
                    "messages": []
                }
            
            if is_toolcall:
                new_sample["tools"] = sample["tools"]
            for dialog in messages:
                if dialog["from"] == 'system':
                    raise ValueError('The system message must be only one!')
                if dialog["from"] ==  'human':
                    role = 'user'
                elif dialog["from"] in ["gpt" , "function_call"]:
                    role = 'assistant'
                else:
                    role = 'tool'
                new_sample['messages'].append(
                    {
                        "role": role,
                        "content": dialog["value"]
                    }
                )
            return new_sample
        else:
            raise ValueError('Input format is not alpaca/sharegpt/openai format! ! Converting them to Standard Openai format is supported currently!')
    else:
        if is_openai_format(sample):
            if sample["messages"][0]["role"]=='system':
                messages = sample["messages"][1:]
                new_messages = [sample["messages"][0]]
            else:
                messages = sample["messages"]
                new_messages = []
            judge_list = [not isinstance(i["content"], List) for i in messages if i["role"]=='user']
            judge_str = [not isinstance(i["content"], str) for i in messages if i["role"]=='user']
            if any(judge_list):
                if any(judge_str):
                    raise ValueError('The type of user input is error!')
                else:
                    exist_tags = {}
                    user_msg = [i for i in messages if i["role"]=='user']
                    if "images" in sample:
                        if len(re.findall(rf'{map_multimodal_tag["image"]}', f'{user_msg}')) != len(sample["images"]):
                            raise ValueError("The number of image tags in user input does not match the number of image addresses provided in the images field.!")
                        exist_tags[map_multimodal_tag["image"]] = 0
                    if "videos" in sample:                  
                        if len(re.findall(rf'{map_multimodal_tag["video"]}', f'{user_msg}')) != len(sample["videos"]):
                            raise ValueError("The number of video tags in user input does not match the number of video addresses provided in the videos field.!")
                        exist_tags[map_multimodal_tag["video"]] = 0
                    if "audios" in sample:     
                        if len(re.findall(rf'{map_multimodal_tag["audio"]}', f'{user_msg}')) != len(sample["audios"]):
                            raise ValueError("The number of audio tags in user input does not match the number of audio addresses provided in the audios field.!")
                        exist_tags[map_multimodal_tag["audio"]] = 0
                    tag_to_type = {v: k for k, v in map_multimodal_tag.items()}
                    for dialog in messages:
                        if dialog["role"] == "system":
                            raise ValueError('The system message must be only one!')
                        if dialog["role"] == "user":
                            split_sentence = split_sentence_by_tags(dialog["content"], list(exist_tags.keys()))
                            new_content = []
                            for modal in split_sentence:
                                if modal in exist_tags:
                                    modal_msg = {"type": tag_to_type[modal], f"{tag_to_type[modal]}": sample[f"{tag_to_type[modal]}s"][exist_tags[modal]]}
                                    if resize_image and tag_to_type[modal] == "image":
                                        modal_msg["resized_width"] = resize_image[0]
                                        modal_msg["resized_height"] = resize_image[1]
                                    elif resize_video and tag_to_type[modal] == "video":
                                        modal_msg["resized_width"] = resize_video[0]
                                        modal_msg["resized_height"] = resize_video[1]
                                    elif fps_video and tag_to_type[modal] == "video":
                                        modal_msg["fps"] = fps_video
                                    new_content.append(modal_msg)
                                    exist_tags[modal]+=1
                                else:
                                    new_content.append({"type": "text", "text": modal})
                            new_messages.append({"role": "user", "content": new_content})
                        else:
                            if is_toolcall and dialog['role'] == 'assistant' and dialog.get('tool_calls'):
                                new_messages.append({"role": dialog['role'], "content": dialog["content"], "too_calls": dialog["tool_calls"]})
                            else:
                                new_messages.append({"role": dialog['role'], "content": dialog["content"]})
                    sample['messages'] = new_messages
                    return sample
            else:
                judge_dict = [k.get("type") not in ("text", "image", "image_url", "video", "audio") for j in messages if j["role"] =='user' for k in j["content"]]
                judge_key = [k.get(k.get("type")) is None for j in messages if j["role"] =='user' for k in j["content"]]
                if any(judge_dict):
                    raise ValueError('The type name must be in ("text", "image", "image_url", "video")!')
                else:
                    if any(judge_key):
                        raise ValueError('The type name must exist as a key in the mapping!')
                    else:
                        return sample
        elif is_sharegpt_format(sample):
            if sample["conversations"][0]["from"]=='system':
                messages = sample["conversations"][1:]
                new_messages = [{"role": "system", "content": sample["conversations"][0]["value"]}]
            else:
                messages = sample["conversations"]
                new_messages = []
            judge_str = [not isinstance(i["value"], str) for i in messages if i["from"] == "human"]
            if any(judge_str):
                raise ValueError('The type of user input is error!')
            else:
                new_sample = deepcopy(sample)
                exist_tags = {}
                user_msg = [i for i in messages if i["from"]=='human']
                if "images" in sample:     
                    if len(re.findall(rf'{map_multimodal_tag["image"]}', f'{user_msg}')) != len(sample["images"]):
                        raise ValueError("The number of image tags in user input does not match the number of image addresses provided in the images field.!")
                    exist_tags[map_multimodal_tag["image"]] = 0
                if "videos" in sample:                  
                    if len(re.findall(rf'{map_multimodal_tag["video"]}', f'{user_msg}')) != len(sample["videos"]):
                        raise ValueError("The number of video tags in user input does not match the number of video addresses provided in the videos field.!")
                    exist_tags[map_multimodal_tag["video"]] = 0
                if "audios" in sample:     
                    if len(re.findall(rf'{map_multimodal_tag["audio"]}', f'{user_msg}')) != len(sample["audios"]):
                        raise ValueError("The number of audio tags in user input does not match the number of audio addresses provided in the audios field.!")
                    exist_tags[map_multimodal_tag["audio"]] = 0
                tag_to_type = {v: k for k, v in map_multimodal_tag.items()}
                for dialog in messages:
                    if dialog["from"] == "system":
                        raise ValueError('The system message must be only one!')
                    if dialog["from"] == "human":
                        split_sentence = split_sentence_by_tags(dialog["value"], list(exist_tags.keys()))
                        new_content = []
                        for modal in split_sentence:
                            if modal in exist_tags:
                                modal_msg = {"type": tag_to_type[modal], f"{tag_to_type[modal]}": sample[f"{tag_to_type[modal]}s"][exist_tags[modal]]}
                                if resize_image and tag_to_type[modal] == "image":
                                    modal_msg["resized_width"] = resize_image[0]
                                    modal_msg["resized_height"] = resize_image[1]
                                elif resize_video and tag_to_type[modal] == "video":
                                    modal_msg["resized_width"] = resize_video[0]
                                    modal_msg["resized_height"] = resize_video[1]
                                elif fps_video and tag_to_type[modal] == "video":
                                    modal_msg["fps"] = fps_video
                                new_content.append(modal_msg)
                                exist_tags[modal]+=1
                            else:
                                new_content.append({"type": "text", "text": modal})
                        new_messages.append({"role": "user", "content": new_content})
                    else:
                        if dialog["from"] in ["gpt" , "function_call"]:
                            role = 'assistant'
                        else:
                            role = 'tool'
                        new_messages.append({"role": role, "content": dialog["value"]})
                new_sample['messages'] = new_messages
                del new_sample["conversations"]
                return new_sample
        else:
            raise ValueError('Input format is not alpaca/sharegpt/openai format! Converting them to standard openai format is supported currently!')

if __name__ == "__main__":
    from pprint import pprint
    import json
    json_file_path = "/home/gaowenjin/code/LLaMA-Factory/data/glaive_toolcall_en_demo.json"
    # json_file_path = "/home/gaowenjin/code/LLaMA-Factory/data/mllm_video_audio_demo.json"
    # json_file_path = "/home/gaowenjin/code/LLaMA-Factory/data/alpaca_zh_demo.json"
    # json_file_path = "/home/gaowenjin/code/LLaMA-Factory/data/kto_en_demo.json"
    with open(json_file_path,'r',encoding="utf-8") as ff:
        data = json.load(ff)
    
    print('multimodal:',is_multimodal_format(data[0]))
    print('toolcall:',is_toolcall_format(data[0]))
    print('alpaca:',is_alpaca_format(data[0]))
    print('sharegpt:',is_sharegpt_format(data[0]))
    print('openai:',is_openai_format(data[0]))
    pprint(convert_to_standard_openai_format(data[0]))
    
    from transformers import AutoProcessor,AutoTokenizer, Qwen2_5_VLForConditionalGeneration
    import torch
    from qwen_vl_utils import process_vision_info
    processor = AutoProcessor.from_pretrained("/home/gaowenjin/models/Qwen2.5-VL-3B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("/home/gaowenjin/models/Qwen2.5-VL-3B-Instruct")
    print(process_vision_info(convert_to_standard_openai_format(data[0])["messages"]))
    print(processor.apply_chat_template(convert_to_standard_openai_format(data[0])["messages"], tokenize=False, add_generation_prompt=True))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/home/gaowenjin/code/LLaMA-Factory/saves/Qwen2.5-1.5B-Instruct/full/train_2025-10-11-11-45-52/training_loss.png", "resized_height": 256, "resized_width": 256},
                {"type": "text", "text": "请描述这张图片的内容。"},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # print(inputs)
    inputs = {key: value.tolist() for key, value in inputs.items()}
    
    # 构造目标输出
    response = tokenizer("It is training loss", add_special_tokens=False)
    input_ids = inputs["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = inputs["attention_mask"][0] + response["attention_mask"] + [1]
    labels = [-100] * len(inputs["input_ids"][0]) + response["input_ids"] + [tokenizer.pad_token_id]

    # print({
    #     "input_ids": torch.tensor(input_ids),
    #     "attention_mask": torch.tensor(attention_mask),
    #     "labels": torch.tensor(labels),
    #     "pixel_values": torch.tensor(inputs["pixel_values"]),
    #     "image_grid_thw": torch.tensor(inputs["image_grid_thw"]).squeeze(0)
    # })
    print(torch.tensor(inputs["pixel_values"]).shape)
    print(torch.tensor(input_ids))