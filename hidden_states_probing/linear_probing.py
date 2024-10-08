# Encoding: UTF-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from tqdm import tqdm
import pandas as pd
from transformers import TrainingArguments, Trainer
# Import CrossEntropyLoss
from torch.nn import CrossEntropyLoss
from functools import partial
import json
import pathlib
from datasets import Dataset, load_dataset, concatenate_datasets

output_dir = "./linear_probing"

torch.cuda.empty_cache()
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="no",
    eval_steps=10000,
    report_to="tensorboard",
    logging_strategy='steps',
    logging_steps=10,
    logging_dir=os.path.join(output_dir, 'logs'),
    save_strategy='steps',
    save_steps=10000,
    num_train_epochs=8,
    remove_unused_columns=False,
    ignore_data_skip=True,
    save_only_model=True,

    optim="adamw_torch",  # 'paged_adamw_8bit',#
    weight_decay=0,

    lr_scheduler_type="constant_with_warmup",  # "linear",  #
    warmup_ratio=0.05,
    learning_rate=1e-4,
    per_device_train_batch_size=200,
    max_grad_norm=1.0,

    # max_steps=1,
    auto_find_batch_size=False,
    load_best_model_at_end=False,
    dataloader_pin_memory=False,

    seed=1,
)


def data_collator(features):
    # features contain hidden_states and labels
    batch = {}
    batch["hidden_states"] = torch.tensor([feature["hidden_states"] for feature in features]).float().cuda()
    batch["labels"] = torch.tensor([feature["labels"] for feature in features]).long().cuda()
    return batch


class LinearProbing(torch.nn.Module):
    def __init__(self, hidden_size, output_dim):
        super(LinearProbing, self).__init__()
        # self.linear1=torch.nn.Linear(hidden_size,hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_dim)

    def forward(self, hidden_states, labels=None):
        # logits=self.linear2(self.linear1(hidden_states))
        logits = self.linear2(hidden_states)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits, "loss": None}


def evaluate(model, eval_dataset):
    model.eval()
    # For each data, linear gives classification results, calculate accuracy
    pred_list = []

    for i in tqdm(range(len(eval_dataset))):
        hidden_states = data_collator([eval_dataset[i]])["hidden_states"]
        with torch.no_grad():
            logits = model(hidden_states)["logits"]
        pred = torch.argmax(logits[0], dim=-1).item()
        pred_list.append(pred)

    # Calculate accuracy
    labels = [int(eval_dataset[i]["labels"]) for i in range(len(eval_dataset))]
    correct_num = sum([1 for i in range(len(labels)) if labels[i] == pred_list[i]])
    acc = correct_num / len(labels)
    return acc


if __name__ == '__main__':

    # "v2k": Direct value-to-key retrieval. The value is given and the model needs to retrieve the corresponding key.
    # "logic": logic-based KV retrieval. All the values are in range 0-9.
    # "multi_match": multi-match KV retrieval. Here is 3-match.

    task_type="logic" # "multi_match" "v2k"

    probing_dataset_path = "./hidden_states_probing_data/{}/states_phi3.jsonl".format(task_type)
    hidden_size = 3072  # hidden_size of phi3
    output_dim = 10

    if task_type == "multi_match":
        #if task is multi_match, you need to identify which token's hidden states are used as the input feature for probing, and which key is used as the label for probing
        label_key_index = 0   # the index of the key which is used as the label for probing
        hidden_token_index = 0  # the index of the token whose hidden states are used as the input feature for probing
        save_path = "./probing_acc_phi3/"+"{}_input_{}_label_{}.json".format(task_type,hidden_token_index,label_key_index)

    else:
        label_key_index = None
        hidden_token_index = None
        save_path = "./probing_acc_phi3/" + "{}.json".format(task_type)

    pathlib.Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

    print("Reading dataset:", probing_dataset_path)
    df = pd.read_json(probing_dataset_path, orient="records", lines=True, dtype=False)
    # Rename hidden_states_last_token to hidden_states
    if "hidden_states_last_token" in df.columns:
        df.rename(columns={"hidden_states_last_token": "hidden_states"}, inplace=True)

    # Select the first digit of the key to be predicted as the label
    if label_key_index is not None:
        df["labels"] = df["gold_keys"].swifter.apply(lambda x: int(x[label_key_index][0]))
    else:
        df["labels"] = df["gold_keys"].swifter.apply(lambda x: int(x[0]))

    # Split hidden_states into different layers, forming 32 dataframes
    results = {}
    for layer in range(32):
        print("layer:", layer)
        df_layer = df.copy()
        if hidden_token_index is not None:
            df_layer["hidden_states"] = df_layer["hidden_states"].swifter.apply(lambda x: x[layer][hidden_token_index])
        else:
            df_layer["hidden_states"] = df_layer["hidden_states"].swifter.apply(lambda x: x[layer])
        dataset = Dataset.from_pandas(df_layer)

        # Split test set
        dataset_dict = dataset.train_test_split(test_size=0.2, seed=1)
        train_dataset = dataset_dict["train"]
        test_dataset = dataset_dict["test"]

        print("Number of training samples:", len(train_dataset))

        # Initialize a torch model, one linear layer
        model = LinearProbing(hidden_size, output_dim)
        model.requires_grad_()
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator,
        )
        # Start training
        trainer.train()

        # Evaluate
        acc = evaluate(model, test_dataset)
        print("layer:", layer, "acc:", acc)
        results[str(layer)] = round(acc, 4)

    # Save
    with open(save_path, "w") as f:
        json.dump(results, f)

    print("linear probing accuracy of each layer:", results)
    print("save accuracy to:", save_path)