import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import pandas as pd
from tqdm import tqdm
import pathlib
import tiktoken
from openai import OpenAI

encoding = tiktoken.encoding_for_model('gpt-4')

from metric import eval_df_logic,eval_df_simple,eval_df_multi_match,eval_df_multi_step
METRIC_MAP={
    "logic":eval_df_logic,
    "simple":eval_df_simple,
    "multi_match":eval_df_multi_match,
    "multi_step":eval_df_multi_step
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


def run_generation(df, client, test_model_name, max_tokens, temperature):

    for i in tqdm(range(0, len(df))):
        # if "answer" already exists, skip
        if "answer" in df.columns and not pd.isna(df.loc[i, "answer"]):
            continue
        prompt = df.loc[i, "prompt"]
        prompt = prompt.replace(r"{ answer }", r"{answer}")

        pred = \
        get_response([prompt], client, model_name=test_model_name, max_tokens=max_tokens, temperature=temperature)[0]
        df.loc[i, "answer"] = pred

        # if i < 1:
        #     print("prompt长度：", len(encoding.encode(prompt)))
        #     print("\n\nOutput:", pred)

    return df


if __name__ == '__main__':

    # choose task and eval function
    task="logic"
    eval_df=METRIC_MAP[task]

    # choose datasets to test
    df_path_list = [
        "./logic-based/data_kv/logic_kv_4.jsonl",
        "./logic-based/data_kv/logic_kv_10.jsonl"
    ]

    # choose model to test
    test_model_name = "llama3.1-70b"#  "gpt-4o-2024-08-06"  # "phi3.5" #

    # use vllm server
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:5000/v1")

    for df_path in df_path_list:
        df = pd.read_json(df_path, lines=True, dtype=False)

        df = run_generation(df, client, test_model_name, max_tokens=512, temperature=0.8)

        df = eval_df(df, task="kv")

        save_path = "./generations_of_{}/".format(test_model_name) + df_path
        pathlib.Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        df.to_json(save_path, orient='records', lines=True)
        print("prompt len: ", len(encoding.encode(df.loc[0, "prompt"])))
        print("save to: ", save_path)
        print("\n\n\n")
