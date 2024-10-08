import json
import random

def generate_kv_list(kv_num=100, use_digit=True, values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    # Generate kv_num unique strings, each consisting of random letters
    import string
    len_str = 10
    str_list = []
    str_feature_list = []
    for i in range(kv_num):
        while True:
            if not use_digit:
                new_str = ''.join(random.sample(string.ascii_lowercase, len_str))
                # The first two characters of the new str must not be repeated
                no_repeat_chars = 2
                if kv_num > 10 and kv_num < 100:
                    no_repeat_chars = 2
                elif kv_num >= 100 and kv_num < 1000:
                    no_repeat_chars = 3
                elif kv_num >= 1000:
                    no_repeat_chars = 4
                if new_str[:no_repeat_chars] not in str_feature_list:
                    str_list.append(new_str)
                    str_feature_list.append(new_str[:no_repeat_chars])
                    break
            else:
                new_str = ''.join(random.sample(string.digits, len_str))
                # The new str must not be repeated
                if new_str not in str_feature_list:
                    str_list.append(new_str)
                    break

    # Generate kv_num single-digit numbers, each between 0 and 5
    num_list = []
    for i in range(kv_num):
        new_num = random.sample(values, 1)[0]
        num_list.append(new_num)

    # Combine strings and numbers into kv pairs
    kv_list = []
    for i in range(kv_num):
        kv_list.append([str_list[i], num_list[i]])

    return kv_list

def kv2dict(kv_list):
    kv_dict = {}
    for kv in kv_list:
        kv_dict[kv[0]] = kv[1]
    return kv_dict

def prompt_v2k_multi(num_kvs=100, gold_key_num=5, gold_key_pos=None, gold_value=0, other_values=[1, 2, 3, 4], concat_answer=False, question_format=False):
    kv_list = generate_kv_list(num_kvs, values=other_values, use_digit=True)
    # Randomly select gold_key_num keys from kv_list and set their value to gold_value
    if gold_key_pos is None:
        gold_keys = random.sample(kv_list, gold_key_num)
    else:
        assert len(gold_key_pos) == gold_key_num, "The length of gold_key_pos should be equal to gold_key_num"
        gold_keys = [kv_list[i] for i in gold_key_pos]
    for key in gold_keys:
        key[1] = gold_value

    if not question_format:
        prompt = "Json data with {} key-value pairs:\n{}\n\nIn the above json data, all the keys whose value is {} are: ".format(num_kvs, json.dumps(kv2dict(kv_list)), gold_value)
        if concat_answer:
            # Enclose each key in quotes and join them with commas
            answer = ", ".join(["\"" + key[0] + "\"" for key in gold_keys])
            prompt += answer
        else:
            prompt += "\""

    elif question_format == "cot":
        prompt = "Json data with {} key-value pairs:\n{}\n\nQuestion: In the above json data, please find all the keys with the value {}. Let's think step by step, and give your final answer (the keys separated by comma and in the order they appear in the context) in format of \"keys: {{ answer }}\"".format(num_kvs, json.dumps(kv2dict(kv_list)), gold_value)
    else:
        prompt = "Json data with {} key-value pairs:\n{}\n\nQuestion: In the above json data, please find all the keys with the value {}. Give your answer (the keys separated by comma and in the order they appear in the context) in format of \"keys: {{ answer }}\"".format(num_kvs, json.dumps(kv2dict(kv_list)), gold_value)
    prompt = prompt.replace(r"{ answer }", r"{answer}")

    return prompt, [key[0] for key in gold_keys], gold_value, kv_list

def prompt_v_range(num_kvs=100, gold_key_num=1, gold_key_pos=[50], gold_value=5, gold_value_bound=(4, 6), other_values=[0, 1, 2, 3, 7, 8, 9], question_format=False):
    # Ensure that no values in other_values are within the gold_value_bound range
    other_values = [num for num in other_values if num < gold_value_bound[0] or num > gold_value_bound[1]]

    kv_list = generate_kv_list(num_kvs, values=other_values, use_digit=True)
    # Randomly select gold_key_num keys from kv_list and set their value to gold_value
    if gold_key_pos is None:
        gold_keys = random.sample(kv_list, gold_key_num)
    else:
        assert len(gold_key_pos) == gold_key_num, "The length of gold_key_pos should be equal to gold_key_num"
        gold_keys = [kv_list[i] for i in gold_key_pos]

    # Set the value of gold_keys to gold_value
    for key in gold_keys:
        key[1] = gold_value

    if not question_format:
        prompt = "Json data with {} key-value pairs:\n{}\n\nIn the above json data, the Key with the Value greater than {} and smaller than {} is: \"".format(num_kvs, json.dumps(kv2dict(kv_list)), gold_value_bound[0], gold_value_bound[1])
    elif question_format == "cot":
        prompt = "Json data with {} key-value pairs:\n{}\n\nQuestion: In the above json data, please find the Key (only one) whose Value (an integer) is greater than {} and smaller than {}. Let's think step by step, and give your final answer (the key) in format of \"key: {{ answer }}\"".format(num_kvs, json.dumps(kv2dict(kv_list)), gold_value_bound[0], gold_value_bound[1])
    else:
        prompt = "Json data with {} key-value pairs:\n{}\n\nQuestion: In the above json data, please find the Key (only one) whose Value (an integer) is greater than {} and smaller than {}. Give your answer (the key) in format of \"key: {{ answer }}\"".format(num_kvs, json.dumps(kv2dict(kv_list)), gold_value_bound[0], gold_value_bound[1])
    prompt = prompt.replace(r"{ answer }", r"{answer}")

    return prompt, [key[0] for key in gold_keys], gold_value, kv_list

def prompt_v2k_direct(num_kvs=100, gold_value=5, other_values=[0, 1, 2, 3, 7, 8, 9], question_format=False):
    kv_list = generate_kv_list(num_kvs, values=other_values, use_digit=True)
    # Ensure gold_value is not in other_values
    if gold_value in other_values:
        other_values.remove(gold_value)
    # Randomly select one key from kv_list and set its value to gold_value
    gold_kv = kv_list[num_kvs // 2]
    gold_kv[1] = gold_value
    gold_key = gold_kv[0]

    if not question_format:
        prompt = "Json data with {} key-value pairs:\n{}\n\nIn the above json data, the Key whose Value is {} is: \"".format(num_kvs, json.dumps(kv2dict(kv_list)), gold_value)
    elif question_format == "cot":
        prompt = "Json data with {} key-value pairs:\n{}\n\nQuestion: In the above json data, please find the Key (only one) whose Value is {}. Let's think step by step, and give your final answer (the key) in format of \"key: {{ answer }}\"".format(num_kvs, json.dumps(kv2dict(kv_list)), gold_value)
    else:
        prompt = "Json data with {} key-value pairs:\n{}\n\nQuestion: In the above json data, please find the Key (only one) whose Value is {}. Give your answer (the key) in format of \"key: {{ answer }}\"".format(num_kvs, json.dumps(kv2dict(kv_list)), gold_value)
    prompt = prompt.replace(r"{ answer }", r"{answer}")

    return prompt, gold_key, gold_value, kv_list

def prompt_chain_concat_value(num_kvs=100, add_values=[0, 1, 2, 3], other_values=list(range(0, 10)), question_format=True):

    kv_list = generate_kv_list(num_kvs, values=other_values, use_digit=True)

    # Select len(add_values) + 1 keys and set their values to add_values sequentially
    gold_keys = random.sample(kv_list, len(add_values) + 1)
    for i in range(len(add_values)):
        gold_keys[i][1] = add_values[i]
    # Set the first n characters of the last key to the string concatenation of add_values
    gold_key_str = gold_keys[-1][0]
    gold_key_str = "".join([str(i) for i in add_values]) + gold_key_str[len(add_values):]

    gold_keys[-1][0] = gold_key_str
    # Ensure gold_key_str is unique in the entire kv_list
    gold_key_str_count = [kv[0] for kv in kv_list].count(gold_key_str)
    if gold_key_str_count > 1:
        raise ValueError("The gold_key_str should be unique in the kv_list")

    gold_value = gold_keys[-1][1]

    query_keys = ", ".join(["\"" + key[0] + "\"" for key in gold_keys[:-1]])

    context = "Json dictionary with {} Key-Value pairs:\n\n{}".format(num_kvs, json.dumps(kv2dict(kv_list)))

    # The task is: sequentially retrieve len(add_values) keys, sum their values, and then retrieve the key whose value equals the sum
    if not question_format:
        raise ValueError("The question_format should be specified")
    elif question_format == "cot":
        prompt = context + "\n\nQuestion: In the above json data, please find the value (you need to search it in the Json dictionary) of the Key. The Key is the string S. \nS is the sequential concatenation of A and B. \nA is the sequential concatenation of the corresponding values (you need to search it in the Json dictionary) of the keys {}. When concatenating, each value is seen as a character.\nB is a string \"{}\". \nLet's think step by step, and give your final answer (the key and the value) in format of \"key:{{ answer }} value:{{ answer }}\"".format(query_keys, gold_key_str[len(add_values):])
    else:
        prompt = context + "\n\nQuestion: In the above json data, please find the value (you need to search it in the Json dictionary) of the Key. The Key is the string S. \nS is the sequential concatenation of A and B. \nA is the sequential concatenation of the corresponding values (you need to search it in the Json dictionary) of the keys {}. When concatenating, each value is seen as a character.\nB is a string \"{}\". \nGive your answer (the key and the value) in format of \"key:{{ answer }} value:{{ answer }}\"".format(query_keys, gold_key_str[len(add_values):])
    prompt = prompt.replace(r"{ answer }", r"{answer}")

    return prompt, gold_keys[-1][0], gold_value, kv_list