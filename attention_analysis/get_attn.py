# Encoding: UTF-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

# Get the current script's path
current_path = os.path.abspath(__file__)
# Get the parent path of the current script
parent_path = os.path.dirname(current_path)
# Get the grandparent path of the current script
grandfather_path = os.path.dirname(parent_path)

sys.path.append(grandfather_path)
sys.path.append(parent_path)

import transformers
import torch
from transformers import AutoTokenizer, modeling_utils, AutoModelForCausalLM, LlamaForCausalLM
import pandas as pd
from tqdm import tqdm
import pathlib
# Import CrossEntropyLoss
from torch.nn import CrossEntropyLoss
import numpy as np
import random


def get_attn_grad(texts, model, tokenizer, gold_values=None, need_grad=False):
    if need_grad:
        model.config.attn_grad = True
        model.requires_grad_(True)
    else:
        model.config.attn_grad = False
        model.requires_grad_(False)
        model.eval()
    inputs = tokenizer(texts[0], return_tensors="pt", padding=False, truncation=True, max_length=100000).to(model.device)
    input_len_fix = inputs.input_ids.size(1)
    attn_all = [torch.zeros([1, model.config.num_attention_heads, input_len_fix, input_len_fix]).to(model.device) for i in range(model.config.num_hidden_layers)]
    attn_grad_all = [torch.zeros([1, model.config.num_attention_heads, input_len_fix, input_len_fix]).to(model.device) for i in range(model.config.num_hidden_layers)]
    loss_all = 0
    # Encode each text separately
    for i, text in tqdm(enumerate(texts), total=len(texts)):
        inputs = tokenizer(text, return_tensors="pt", padding=False, truncation=True, max_length=1000000).to(model.device)

        input_ids = inputs.input_ids
        input_len = input_ids.size(1)
        print("Input length:", input_len)
        if input_len != input_len_fix:
            print("Input length mismatch:", input_len, input_len_fix)
            continue

        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

        if need_grad:
            # Get logits, take the last token
            logits = outputs.logits[0, -1]
            # Get label
            label_str = gold_values[i][0]
            label = tokenizer.encode(label_str, return_tensors="pt", add_special_tokens=False).to(model.device)[0][1]
            # Calculate loss
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), label.view(-1))
            print("Loss:", loss.item())
            loss_all += loss.item()
            # Calculate gradient
            loss.backward()
        # Get attentions
        attns = outputs.attentions
        if need_grad:
            attns_grad = [attn.grad for attn in attns]
        else:
            # attns_grad are all zeros
            attns_grad = [torch.zeros_like(attn) for attn in attns]
        # Add to attn_all and attn_grad_all
        for j in range(model.config.num_hidden_layers):
            attn_all[j] += attns[j].detach()
            attn_grad_all[j] += attns_grad[j].detach()

        model.zero_grad()

    attn_all = torch.cat(attn_all, dim=0)
    attn_grad_all = torch.cat(attn_grad_all, dim=0)
    return attn_all / len(texts), attn_grad_all / len(texts), loss_all / len(texts)


def match_answer(pred, ref):
    # If ref is a list
    if isinstance(ref, list):
        # Split pred by commas into multiple integers
        try:
            pred_list = [int(i) for i in pred.split(",")]
        except:
            return False
        # If pred_list is equal to ref
        if pred_list == ref:
            return True
        else:
            return False

    # If ref is an integer
    else:
        try:
            int(pred)
        except:
            return False
        if int(pred) == ref:
            return True
        else:
            return False


def main(num_samples=1000, task_type="logic", model_path="/Phi-3.5-mini-instruct"):
    # Load model
    model_name = model_path
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype="auto",
                                                 trust_remote_code=True,
                                                 attn_implementation="eager",  # "flash_attention_2",
                                                 device_map="auto"
                                                 )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Generate prompt
    samples_all = []

    if task_type == "v2k":
        from prompt import prompt_v2k_direct

        for gold_value in [2, 3, 4, 5, 6, 7]:
            gold_value_lower_bound = gold_value - 1
            gold_value_upper_bound = gold_value + 1
            other_values = [i for i in range(10) if i < gold_value_lower_bound or i > gold_value_upper_bound]
            samples = [prompt_v2k_direct(100, gold_value, other_values) for i in tqdm(range(100))]

            samples_all.extend(samples)

    elif task_type == "logic":
        from prompt import prompt_v_range
        for gold_value in [2, 3, 4, 5, 6, 7]:
            gold_value_lower_bound = gold_value - 1
            gold_value_upper_bound = gold_value + 1
            other_values = [i for i in range(10) if i < gold_value_lower_bound or i > gold_value_upper_bound]
            samples = [prompt_v_range(100, 1, [50], gold_value, (gold_value_lower_bound, gold_value_upper_bound), other_values) for i in tqdm(range(100))]
            samples_all.extend(samples)

    elif task_type == "multi_match":
        from prompt import prompt_v2k_multi

        for gold_value in range(10):
            other_values = [i for i in range(10) if i != gold_value]
            samples = [prompt_v2k_multi(100, 3, [45, 50, 55], gold_value, other_values, concat_answer=True) for i in
                       range(num_samples)]
            samples_all.extend(samples)

    else:
        raise ValueError("task_type not supported! we only support 'logic','multi_match','v2k'")

    samples = samples_all

    texts = [sample[0] for sample in samples]
    gold_keys = [sample[2] for sample in samples]
    gold_values = [sample[1] for sample in samples]

    # Get attention
    attn, attn_grad, loss = get_attn_grad(texts, model, tokenizer, gold_values)
    save_dir = "./attention_result/" + task_type + "/"
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Save attention
    attn = attn.cpu().numpy()
    # Average across attention heads
    attn = attn.mean(axis=1)
    print("attn.shape:", attn.shape)
    np.save(save_dir + "attn.npy", attn)
    print("save to: ", save_dir + "attn.npy")
    # Save loss as json
    import json
    with open(save_dir + "loss.json", "w") as f:
        json.dump({"loss": loss}, f)


if __name__ == '__main__':

    # "v2k": Direct value-to-key retrieval. The value is given and the model needs to retrieve the corresponding key.
    # "logic": logic-based KV retrieval. All the values are in range 0-9.
    # "multi_match": multi-match KV retrieval. Here is 3-match.

    task_type = "logic"  # "multi_match" "v2k"
    model_path = "/Phi-3.5-mini-instruct"

    df = main(num_samples=100, task_type=task_type, model_path=model_path)