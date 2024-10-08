import os
import pathlib

import pandas as pd
import tiktoken
from openai import OpenAI
from tqdm import tqdm

encoding = tiktoken.encoding_for_model('gpt-4')

from metric import eval_df_logic,eval_df_simple,eval_df_multi_match,eval_df_multi_step
METRIC_MAP={
    "logic":eval_df_logic,
    "simple_k2v":eval_df_simple,
    "simple_v2k": eval_df_simple,
    "multi_match":eval_df_multi_match,
    "multi_match_last":eval_df_multi_match,
    "multi_step":eval_df_multi_step
}

DATASET_MAP={
    "logic":[
            "./logic-based/data_kv/logic_kv_4.jsonl",
            "./logic-based/data_kv/logic_kv_10.jsonl",
            "./logic-based/data_kv/logic_kv_100.jsonl",
            "./logic-based/data_kv/logic_kv_1000.jsonl",
            "./logic-based/data_student/logic_gpa_resume_4.jsonl",
            "./logic-based/data_student/logic_gpa_resume_10.jsonl",
            "./logic-based/data_student/logic_gpa_resume_100.jsonl",
        ],
    "simple_k2v":[
            "./simple-retrieval/data-kv/simple_k2v_v100_kv_10.jsonl",
            "./simple-retrieval/data-kv/simple_k2v_v100_kv_100.jsonl",
            "./simple-retrieval/data-kv/simple_k2v_v100_kv_1000.jsonl",
            "./simple-retrieval/data-kv/simple_k2v_v100_kv_3000.jsonl",
        ],
    "simple_v2k":[
            "./simple-retrieval/data-kv/simple_v2k_v100_kv_10.jsonl",
            "./simple-retrieval/data-kv/simple_v2k_v100_kv_100.jsonl",
            "./simple-retrieval/data-kv/simple_v2k_v100_kv_1000.jsonl",
            "./simple-retrieval/data-kv/simple_v2k_v100_kv_3000.jsonl",
        ],
    "multi_match":[
            "./multi-matching/data-kv/1_match_kv_10.jsonl",
            "./multi-matching/data-kv/1_match_kv_100.jsonl",
            "./multi-matching/data-kv/1_match_kv_1000.jsonl",

            "./multi-matching/data-kv/5_match_kv_10.jsonl",
            "./multi-matching/data-kv/5_match_kv_100.jsonl",
            "./multi-matching/data-kv/5_match_kv_1000.jsonl",

            "./multi-matching/data-kv/10_match_kv_10.jsonl",
            "./multi-matching/data-kv/10_match_kv_100.jsonl",
            "./multi-matching/data-kv/10_match_kv_1000.jsonl",

            "./multi-matching/data-kv/20_match_kv_100.jsonl",
            "./multi-matching/data-kv/20_match_kv_1000.jsonl",

            "./multi-matching/data-student/1_match_resume_10.jsonl",
            "./multi-matching/data-student/1_match_resume_100.jsonl",

            "./multi-matching/data-student/5_match_resume_10.jsonl",
            "./multi-matching/data-student/5_match_resume_100.jsonl",

            "./multi-matching/data-student/10_match_resume_10.jsonl",
            "./multi-matching/data-student/10_match_resume_100.jsonl",
        ],

    "multi_match_last": [
        "./multi-matching/data-kv-last/3_match_kv_100_only_last.jsonl",
        "./multi-matching/data-kv-last/3_match_kv_1000_only_last.jsonl",
        "./multi-matching/data-kv-last/10_match_kv_100_only_last.jsonl",
        "./multi-matching/data-kv-last/10_match_kv_1000_only_last.jsonl",
        "./multi-matching/data-kv-last/100_match_kv_100_only_last.jsonl",
        "./multi-matching/data-kv-last/100_match_kv_1000_only_last.jsonl",],

    "multi_step":[
        "./multi-step/data-kv/concat_1_kv_10_cot.jsonl",
        "./multi-step/data-kv/concat_1_kv_100_cot.jsonl",
        "./multi-step/data-kv/concat_1_kv_1000_cot.jsonl",
        "./multi-step/data-kv/concat_3_kv_10_cot.jsonl",
        "./multi-step/data-kv/concat_3_kv_100_cot.jsonl",
        "./multi-step/data-kv/concat_3_kv_1000_cot.jsonl",
        "./multi-step/data-kv/concat_5_kv_10_cot.jsonl",
        "./multi-step/data-kv/concat_5_kv_100_cot.jsonl",
        "./multi-step/data-kv/concat_5_kv_1000_cot.jsonl",]

}

def get_response(prompts, client, model_name, max_tokens=512, temperature=0.8):
    answer_list = []
    for prompt in prompts:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                timeout=60,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            print(e)
            answer = ""

        answer_list.append(answer)

    return answer_list


def run_generation(df, client, test_model_name, max_tokens, temperature,suffix=None):

    for i in tqdm(range(0, len(df))):
        # if "answer" already exists, skip
        if "answer" in df.columns and not pd.isna(df.loc[i, "answer"]):
            continue
        prompt = df.loc[i, "prompt"]
        prompt = prompt.replace(r"{ answer }", r"{answer}")
        if suffix is not None:
            prompt=prompt+" "+suffix

        pred = \
        get_response([prompt], client, model_name=test_model_name, max_tokens=max_tokens, temperature=temperature)[0]
        df.loc[i, "answer"] = pred


    return df


if __name__ == '__main__':

    # choose one of the tasks
    # "simple_k2v": Direct key-to-value retrieval. The key is given and the model needs to retrieve the corresponding value.
    # "simple_v2k": Direct value-to-key retrieval. The value is given and the model needs to retrieve the corresponding key.
    # "logic": logic-based KV retrieval. All the values are in range 0-9. We give the range of the value and the model needs to retrieve the corresponding key.
    # "multi_step": multi-step KV retrieval. The model needs to retrieve multiple values with multiple queries.
    # "multi_match": multi-match KV retrieval. The value is given and the model needs to retrieve multiple corresponding keys.
    # "multi_match_last": multi-match KV retrieval. The value is given and the model needs to retrieve multiple corresponding keys. The other gold keys are already given in the prompt, except the last one.

    task_type="logic" # "logic","multi_match","multi_step","simple_k2v","simple_v2k"

    # choose cot_prompt
    # None: default prompt, let the model give the answer directly
    # "cot": add CoT prompt, let hte model 'think step by step'
    # "one-by-one": add one-by-one prompt, let the model 'examine every item one by one'

    cot_prompt=None  #None, "cot", "one-by-one"

    # choose model to test. if use local VLLM server, you should first run launch_vllm_server.sh
    test_model_name = "llama3.1-70b"# "gpt-4o-2024-08-06"  #  "phi3.5" #

    # use openai server
    #client = OpenAI(api_key="your_api_key", base_url="https://api.openai.com/v1")

    # use vllm server
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:5000/v1")


    # choose datasets to test
    df_path_list=DATASET_MAP[task_type]
    eval_df=METRIC_MAP[task_type]


    for df_path in df_path_list:
        df = pd.read_json(df_path, lines=True, dtype=False)
        print("evaluating: ", df_path)

        if cot_prompt==None:
            suffix=None
        elif cot_prompt=="cot":
            suffix = " Let's think step by step before giving the final answer, but you cannot check one by one."
        elif cot_prompt=="one-by-one":
            suffix = "You should first examine every item one by one to give the judgement (yes/no) whether it meet the requirement, and then summarize to give the final answer."
        else:
            raise ValueError("Unknown cot_prompt")

        # get model responses
        df = run_generation(df, client, test_model_name, max_tokens=512, temperature=0.8,suffix=suffix)

        # evaluate model responses to get accuracy
        if "k2v" in df_path:
            data_type="k2v"
        elif "v2k" in df_path:
            data_type="v2k"
        elif "resume" in df_path:
            data_type="resume"
        elif "last" in df_path:
            data_type="last_key"
        else:
            data_type="kv"
        df = eval_df(df, task=data_type)

        # save responses
        save_path = "./responses_of_{}/".format(test_model_name) + df_path
        pathlib.Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        df.to_json(save_path, orient='records', lines=True)
        print("prompt len: ", len(encoding.encode(df.loc[0, "prompt"])))
        print("save to: ", save_path)
        print("\n\n\n")
