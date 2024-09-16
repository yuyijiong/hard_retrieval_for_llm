import re

import re

def eval_df_logic(df, task="kv"):
    def match_kv(pred: str, ref: str):
        ref = str(ref).strip()
        # Extract the content after 'key:' in pred, stopping at the first occurrence of \n or end of string
        pred_key = re.findall(r"key:(.+?)(?=\n|$)", pred, flags=re.DOTALL | re.IGNORECASE)
        if not pred_key:
            # Find the first occurrence of a sequence of 10 or more digits
            pred_key = re.findall(r"\d{10,}", pred)
        if not pred_key:
            return False, ""
        pred_key = pred_key[-1].replace("\"", "").replace(" ", "").replace("{", "").replace("}", "").strip()
        # If it contains a colon, only take the content before the first colon
        if ":" in pred_key:
            pred_key = pred_key.split(":")[0].strip()

        # Determine correctness
        if pred_key == ref:
            return True, pred_key
        else:
            return False, pred_key

    def match_student(pred: str, ref: str):
        ref = str(ref).strip()
        # Extract the content after 'name:' in pred, stopping at the first occurrence of \n or end of string
        pred_key = re.findall(r"name:(.+?)(?=\n|$)", pred, flags=re.DOTALL | re.IGNORECASE)
        if not pred_key:
            # Find the content before the first colon
            pred_key = pred.split(":")[0].strip()
        else:
            pred_key = pred_key[0]

        if not pred_key:
            return False, ""
        pred_key = pred_key.replace("\"", "").replace("{", "").replace("}", "").strip()
        # If it contains a colon, only take the content before the first colon
        if ":" in pred_key:
            pred_key = pred_key.split(":")[0].strip()

        # Determine correctness
        if pred_key.lower() == ref.lower():
            return True, pred_key
        else:
            return False, pred_key

    # Convert the 'gold_keys' attribute of df to string. Ensure the length is 10, padding with 0s if necessary
    df["gold_keys"] = df["gold_keys"].apply(lambda x: str(x))
    df["gold_keys"] = df["gold_keys"].apply(lambda x: x.zfill(10))

    if task == "kv":
        # Determine accuracy
        df["correct"] = df.apply(lambda x: match_kv(x["answer"], x["gold_keys"])[0], axis=1)
        df["pred_key"] = df.apply(lambda x: match_kv(x["answer"], x["gold_keys"])[1], axis=1)
    else:
        # Determine accuracy
        df["correct"] = df.apply(lambda x: match_student(x["answer"], x["gold_keys"])[0], axis=1)
        df["pred_key"] = df.apply(lambda x: match_student(x["answer"], x["gold_keys"])[1], axis=1)

    print("Accuracy:", df["correct"].mean())

    return df


def eval_df_multi_match(df, task="kv"):
    def match_kv(pred: str, ref: list):
        if isinstance(ref, str):
            ref = [ref]

        # Ensure each element in ref is an integer
        ref = [int(i) for i in ref]

        pred = re.findall(r"keys:(.+?)(?=\n|$)", pred, flags=re.DOTALL)

        # Split pred by commas into multiple integers
        try:
            pred = pred[0]
            pred_list = pred.split(",")
            pred_list = [i.replace("\"", "").replace(" ", "").replace("{", "").replace("}", "").strip() for i in
                         pred_list]
            # If it contains a colon, only take the part before the first colon
            pred_list = [i.split(":")[0] for i in pred_list]
            # Discard if less than 10 digits
            pred_list = [i for i in pred_list if len(i) >= 10]
            pred_list = [int(i) for i in pred_list]
        except:
            return False, []
        # If pred_list and ref are equal (order does not matter)

        # Determine error type: over-selection, under-selection, or wrong selection
        if set(pred_list) == set(ref):
            return "Correct", pred_list
        elif set(pred_list).issubset(set(ref)):
            return "Under-selected", pred_list
        elif set(ref).issubset(set(pred_list)):
            return "Over-selected", pred_list
        else:
            return "Wrong selection", pred_list

    def match_student(pred: str, ref: list):
        if isinstance(ref, str):
            ref = [ref]

        # Ensure each element in ref is a string
        ref = [str(i) for i in ref]

        pred_keys = re.findall(r"names:(.+?)(?=\n|$)", pred, flags=re.DOTALL | re.IGNORECASE)
        if not pred_keys:
            pred_keys = re.findall(r"answer:(.+?)(?=\n|$)", pred, flags=re.DOTALL | re.IGNORECASE)
        if not pred_keys:
            # Find content within {}
            pred_keys = re.findall(r"\{(.+?)\}", pred, flags=re.DOTALL | re.IGNORECASE)
        if not pred_keys:
            # If it contains a space followed by an uppercase letter, consider the entire string as pred_keys
            if_name = re.findall(r"(\S+[A-Z]+)", pred, flags=re.DOTALL | re.IGNORECASE)
            if if_name:
                pred_keys = [pred]
            else:
                pred_keys = []

        # Split pred by commas into multiple strings
        try:
            pred = pred_keys[0]
            pred_list = pred.split(",")
            pred_list = [i.replace("\"", "").replace("{", "").replace("}", "").strip() for i in pred_list]
            # If it contains a colon, only take the part before the first colon
            pred_list = [i.split(":")[0] for i in pred_list]

        except:
            return False, []

        # Convert all names in pred_list and ref to lowercase
        pred_list = [i.lower() for i in pred_list]
        ref = [i.lower() for i in ref]

        # Determine error type: over-selection, under-selection, or wrong selection
        if set(pred_list) == set(ref):
            return "Correct", pred_list
        elif set(pred_list).issubset(set(ref)):
            return "Under-selected", pred_list
        elif set(ref).issubset(set(pred_list)):
            return "Over-selected", pred_list
        else:
            return "Wrong selection", pred_list

    if task == "kv":
        # Determine accuracy
        df["correct"] = df.apply(lambda x: match_kv(x["answer"], x["gold_keys"])[0], axis=1)
        df["pred_keys"] = df.apply(lambda x: match_kv(x["answer"], x["gold_keys"])[1], axis=1)
    else:
        # Determine accuracy
        df["correct"] = df.apply(lambda x: match_student(x["answer"], x["gold_keys"])[0], axis=1)
        df["pred_keys"] = df.apply(lambda x: match_student(x["answer"], x["gold_keys"])[1], axis=1)
    # Calculate the proportion of over-selection, under-selection, wrong selection, correct, and no answer
    over_select = df['correct'].apply(lambda x: 1 if x == "Over-selected" else 0).mean()
    less_select = df['correct'].apply(lambda x: 1 if x == "Under-selected" else 0).mean()
    wrong_select = df['correct'].apply(lambda x: 1 if x == "Wrong selection" else 0).mean()
    correct = df['correct'].apply(lambda x: 1 if x == "Correct" else 0).mean()
    none_answer = df['correct'].apply(lambda x: 1 if x == False else 0).mean()
    print("Correct:", correct, "Over-selected:", over_select, "Under-selected:", less_select, "Wrong selection:", wrong_select, "No answer:", none_answer)

    return df


def eval_df_multi_step(df, task="kv"):
    def match_kv(pred: str, ref: str, type="v"):

        def get_v(pred: str):
            # Find the first numeric string after 'value:'
            pred_key = re.findall(r"value:[^\d]*(\d+)", pred, flags=re.DOTALL | re.IGNORECASE)

            if not pred_key:
                # Find the last numeric string in pred
                pred_key = re.findall(r"\d+", pred)
                pred_key = [pred_key[-1]] if pred_key else ""

            if not pred_key:
                return False, ""

            pred_key = pred_key[-1]
            return pred_key

        def get_key(pred: str):
            # Find the first numeric string after 'key:'
            pred_key = re.findall(r"key:[^\d]*(\d+)", pred, flags=re.DOTALL | re.IGNORECASE)

            if not pred_key:
                # Find all numeric strings of length 10 in pred, taking the last one
                pred_key = re.findall(r"\d{10}", pred)
                pred_key = [pred_key[-1]] if pred_key else ""

            if not pred_key:
                return ""

            pred_key = pred_key[-1]
            return pred_key

        ref = str(ref).strip()

        if type == "v":
            pred_key = get_v(pred)
        else:
            pred_key = get_key(pred)

        # Determine correctness
        if pred_key == ref:
            return True, pred_key
        else:
            return False, pred_key

    # Convert the 'gold_keys' attribute of df to string. Ensure the length is 10, padding with 0s if necessary
    df["gold_keys"] = df["gold_keys"].apply(lambda x: str(x))
    df["gold_keys"] = df["gold_keys"].apply(lambda x: x.zfill(10))

    if task == "kv":
        # Determine value accuracy
        df["value_correct"] = df.apply(lambda x: match_kv(x["answer"], x["gold_values"], type="v")[0], axis=1)
        df["pred_value"] = df.apply(lambda x: match_kv(x["answer"], x["gold_values"], type="v")[1], axis=1)
        # Determine key accuracy
        df["key_correct"] = df.apply(lambda x: match_kv(x["answer"], x["gold_keys"], type="k")[0], axis=1)
        df["pred_key"] = df.apply(lambda x: match_kv(x["answer"], x["gold_keys"], type="k")[1], axis=1)

    else:
        # Determine accuracy
        raise NotImplementedError("Task not implemented")

    print("Key accuracy:", df["key_correct"].mean())
    print("Value accuracy:", df["value_correct"].mean())

    return df


def eval_df_simple(df, task="kv"):
    def match_kv(pred: str, ref: str):
        if isinstance(ref, list):
            ref = ref[0]
        ref = str(ref).strip()
        # Extract the content after 'value:' in pred, stopping at the first occurrence of \n or end of string
        pred_key = re.findall(r"value:(.+?)(?=\n|$)", pred, flags=re.DOTALL | re.IGNORECASE)
        if not pred_key:
            # Find the last numeric string in pred
            pred_key = re.findall(r"\d+", pred)
            pred_key = [pred_key[-1]] if pred_key else ""
        if not pred_key:
            return False, ""
        pred_key = pred_key[0].replace("\"", "").replace(" ", "").replace("{", "").replace("}", "").strip()
        # If it contains a colon, only take the content after the first colon
        if ":" in pred_key:
            pred_key = pred_key.split(":")[-1].strip()

        # Determine correctness
        if pred_key == ref:
            return True, pred_key
        else:
            return False, pred_key


    def match_kv_v2k(pred: str, ref: str):
        if isinstance(ref, list):
            ref = ref[0]
        ref = str(ref).strip()
        # Extract the content after 'key:' in pred, stopping at the first occurrence of \n or end of string
        pred_key = re.findall(r"key:(.+?)(?=\n|$)", pred, flags=re.DOTALL | re.IGNORECASE)
        if not pred_key:
            # Find the last sequence of 10 or more digits in pred
            pred_key = re.findall(r"\d{10,}", pred)
            pred_key = [pred_key[-1]] if pred_key else ""
        if not pred_key:
            return False, ""
        pred_key = pred_key[-1].replace("\"", "").replace(" ", "").replace("{", "").replace("}", "").strip()
        # If it contains a colon, only take the content before the first colon
        if ":" in pred_key:
            pred_key = pred_key.split(":")[0].strip()

        # Determine correctness
        if pred_key == ref:
            return True, pred_key
        else:
            return False, pred_key

    if task == "kv":
        # Determine accuracy
        df["correct"] = df.apply(lambda x: match_kv(x["answer"], x["gold_values"])[0], axis=1)
        df["pred"] = df.apply(lambda x: match_kv(x["answer"], x["gold_values"])[1], axis=1)
    elif task == "v2k":
        # Convert the 'gold_keys' attribute of df to string. Ensure the length is 10, padding with 0s if necessary
        df["gold_keys"] = df["gold_keys"].apply(lambda x: str(x))
        df["gold_keys"] = df["gold_keys"].apply(lambda x: x.zfill(10))
        # Determine accuracy
        df["correct"] = df.apply(lambda x: match_kv_v2k(x["answer"], x["gold_keys"])[0], axis=1)
        df["pred"] = df.apply(lambda x: match_kv_v2k(x["answer"], x["gold_keys"])[1], axis=1)
    else:
        raise NotImplementedError("Task not implemented")

    print("Accuracy:", df["correct"].mean())

    return df