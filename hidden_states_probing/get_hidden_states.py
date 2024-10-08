
# Encoding: UTF-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

sys.path.append('../')
sys.path.append('../../')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import pathlib
import random

random.seed(0)


def get_hidden_state(texts, model, tokenizer, layer: int = None, token_num: int = 1, generate_tokens: int = 1,
                     gold_values=None, token_index=None):
    hidden_states_last_token_list = []
    pred_list = []
    # Encode each text separately
    for i, text in tqdm(enumerate(texts), total=len(texts)):
        inputs = tokenizer(text, return_tensors="pt", padding=False, truncation=True, max_length=100000).to(
            model.device)

        input_ids = inputs.input_ids
        input_len = input_ids.size(1)
        print("Input length:", input_len)
        with torch.no_grad():
            output = model.generate(input_ids=input_ids,
                                    max_new_tokens=generate_tokens,
                                    do_sample=False,
                                    output_hidden_states=True,
                                    return_dict_in_generate=True,
                                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n")[-1]],
                                    use_cache=False if generate_tokens == 1 else True)
        pred_str = tokenizer.decode(output["sequences"][0][input_len:]).strip()
        pred_str = pred_str.replace("and", ",")

        # Get prefill hidden_states
        hidden_states = output.hidden_states[0]
        # Get the last token of hidden_state for each layer
        if token_index is None:
            if layer is None:
                hidden_states_last_token = [hidden_state[0, -token_num:].tolist() for hidden_state in hidden_states[1:]]
            else:
                hidden_states_last_token = hidden_states[layer + 1][0, -token_num:].tolist()
        else:
            if layer is None:
                hidden_states_last_token = [hidden_state[0, token_index:token_index + token_num].tolist() for
                                            hidden_state in hidden_states[1:]]
            else:
                hidden_states_last_token = hidden_states[layer + 1][0, token_index:token_index + token_num].tolist()

        hidden_states_last_token_list.append(hidden_states_last_token)

        pred_list.append(pred_str)
        print("pred:", pred_str, "gold:", gold_values[i])

    return hidden_states_last_token_list, pred_list


def match_answer(pred: str, ref):
    pred = str(pred).split("\n")[0].strip()
    # If ref is a list
    if isinstance(ref, list):
        # Each element in ref is an integer
        ref = [int(i) for i in ref]

        # Split pred into multiple integers by comma
        try:
            pred_list = pred.split(",")
            pred_list = [i.strip().strip("\'").strip("\"") for i in pred_list]
            pred_list = [int(i) for i in pred_list]
        except:
            return False
        # If pred_list and ref are equal (order does not matter)

        # Determine the type of error, whether it is multiple choice, missing choice, or wrong choice
        if set(pred_list) == set(ref):
            return True
        elif set(pred_list).issubset(set(ref)):
            return "Missing choice"
        elif set(ref).issubset(set(pred_list)):
            return "Multiple choice"
        else:
            return "Wrong choice"

    # If ref is a string
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
                                                 attn_implementation="flash_attention_2",  # "eager",#
                                                 device_map="auto"
                                                 )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    samples_all = []

    # Multi-match task
    if task_type == "multi_match":
        from ..prompt import prompt_v2k_multi
        for gold_value in range(10):
            other_values = [i for i in range(10) if i != gold_value]
            samples = [prompt_v2k_multi(100, 3, [45, 50, 55], gold_value, other_values, concat_answer=True) for i in
                       range(num_samples // 10)]
            samples_all.extend(samples)

    # Logic task
    elif task_type == "logic":
        from ..prompt import prompt_v_range

        for gold_value in [2, 3, 4, 5, 6, 7]:
            gold_value_lower_bound = gold_value - 1
            gold_value_upper_bound = gold_value + 1
            other_values = [i for i in range(10) if i < gold_value_lower_bound or i > gold_value_upper_bound]

            samples = [
                prompt_v_range(10, 1, [5], gold_value, (gold_value_lower_bound, gold_value_upper_bound), other_values)
                for i in tqdm(range(num_samples // 6))]
            samples_all.extend(samples)

    # Direct retrieval
    elif task_type == "v2k":
        from ..prompt import prompt_v2k_direct

        for gold_value in [2, 3, 4, 5, 6, 7]:
            gold_value_lower_bound = gold_value - 1
            gold_value_upper_bound = gold_value + 1
            other_values = [i for i in range(10) if i < gold_value_lower_bound or i > gold_value_upper_bound]
            samples = [prompt_v2k_direct(100, gold_value, other_values) for i in tqdm(range(num_samples // 6))]
            samples_all.extend(samples)

    # Set the layers and token positions to be recorded
    layer = None
    token_num = 1
    # If it is a multi_match task, record the hidden_state of 3 tokens, otherwise only record the hidden_state of the last token
    token_index = [-12, -24, -36] if task_type == "multi_match" else None

    # Get hidden_state
    samples = samples_all
    texts = [sample[0] for sample in samples]
    gold_keys = [sample[1] for sample in samples]
    gold_values = [sample[2] for sample in samples]
    hidden_states_last_token_list, pred_list = get_hidden_state(texts, model, tokenizer, layer=layer,
                                                                token_num=token_num, generate_tokens=1,
                                                                gold_values=gold_keys, token_index=token_index)

    # Combine into df
    df = pd.DataFrame({"text": texts, "gold_keys": gold_keys, "gold_values": gold_values,
                       "hidden_states_last_token": hidden_states_last_token_list, "pred": pred_list})

    # Save df
    file_name = "states_phi3.jsonl"
    save_dir = "./hidden_states_probing_data/{}/".format(task_type)
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    df.to_json(save_dir + file_name, orient="records", lines=True)
    print("Saved to:", save_dir + file_name)
    return df


if __name__ == '__main__':
    df = main(num_samples=2000, task_type="multi_match", model_path="Phi-3.5-mini-instruct")
