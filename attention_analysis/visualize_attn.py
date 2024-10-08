import os
import pathlib

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def get_kv_bound(index, type="k", kv_len=15, prefix_len=13):
    if type == "k":
        if index == 0:
            return (prefix_len + 1, prefix_len + 11)
        else:
            return (prefix_len + 1 + index * kv_len, prefix_len + 11 + index * kv_len)

    elif type == "v":
        if index == 0:
            return (prefix_len + 13, prefix_len + 14)
        else:
            return (prefix_len + 13 + index * kv_len, prefix_len + 14 + index * kv_len)

    elif type == "kv":
        if index == 0:
            return (prefix_len + 1, prefix_len + 16)
        else:
            return (prefix_len + 1 + index * kv_len, prefix_len + 16 + index * kv_len)

    else:
        raise ValueError("Invalid type parameter")

def draw_attn(attn_path, kv_num, query_index, gold_index, label, relative=True, kv="k"):

    kv_len = 15

    prefix_len = 13 if kv_num >= 100 else 12

    k_bounds = [get_kv_bound(i, type="k", prefix_len=prefix_len) for i in range(kv_num)]
    v_bounds = [get_kv_bound(i, type="v", prefix_len=prefix_len) for i in range(kv_num)]

    attn = np.load(attn_path) * 1000

    attn_gold_key_list = []
    attn_gold_v_list = []
    for i in tqdm(range(attn.shape[0])):
        attn_layer = attn[i]
        # Calculate attention to gold key
        attn_gold_key = attn_layer[query_index, k_bounds[gold_index][0]:k_bounds[gold_index][1]].mean().item()
        attn_gold_v = attn_layer[query_index, v_bounds[gold_index][0]:v_bounds[gold_index][1]].mean().item()

        if relative:
            # Calculate attention to each key and then average
            attn_each_key = [attn_layer[query_index, k_bounds[k][0]:k_bounds[k][1]].mean().item() for k in range(kv_num)]
            # Take the top 5
            attn_each_key = sorted(attn_each_key, reverse=True)[:10]
            attn_each_key_mean = np.mean(attn_each_key)
            attn_each_v = [attn_layer[query_index, v_bounds[k][0]:v_bounds[k][1]].mean().item() for k in range(kv_num)]
            # Take the top 5
            attn_each_v = sorted(attn_each_v, reverse=True)[:10]
            attn_each_v_mean = np.mean(attn_each_v)

            attn_gold_key_relative = attn_gold_key / attn_each_key_mean
            attn_gold_v_relative = attn_gold_v / attn_each_v_mean

            attn_gold_key_list.append(attn_gold_key_relative)
            attn_gold_v_list.append(attn_gold_v_relative)
        else:
            attn_gold_key_list.append(attn_gold_key)
            attn_gold_v_list.append(attn_gold_v)

    # plt.figure(figsize=(10,6))
    if kv == "k":
        plt.plot(attn_gold_key_list, marker=".", label=label)
    elif kv == "v":
        plt.plot(attn_gold_v_list, marker=".", label=label)
    else:
        raise ValueError("Invalid kv parameter")
    plt.xlabel("Layer", fontsize=25)
    plt.xticks(range(1, 32, 2))
    # ylabel="Relative Attention" if relative else "Attention ($10^{-3}$)"
    if relative:
        ylabel = "Attn to Gold Key" if kv == "k" else "Attn to Gold Value"
    else:
        ylabel = "Attn to Gold Key ($10^{-3}$)" if kv == "k" else "Attn to Gold Value ($10^{-3}$)"

    plt.ylabel(ylabel, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # y-axis ticks need to be rounded to one decimal place
    import matplotlib.ticker as ticker
    formatter = ticker.FormatStrFormatter('%.1f')
    # plt.axis.set_major_formatter(formatter)

    # plt.rcParams.update({'font.size': 20})
    # plt.legend()


if __name__ == '__main__':
    # "v2k": Direct value-to-key retrieval. The value is given and the model needs to retrieve the corresponding key.
    # "logic": logic-based KV retrieval. All the values are in range 0-9.
    # "multi_match": multi-match KV retrieval. Here is 3-match.

    task_type = "logic"  # "multi_match" "v2k"

    # If true, calculate relative attention; otherwise, calculate absolute attention
    relative = True

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    if task_type == "logic":
        attn_path = "./attention_result/logic/attn.npy"
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)

        kv = "k"
        draw_attn(attn_path, kv_num=10, query_index=-1, gold_index=5, label="logical", relative=relative, kv=kv)

        # Hide x-axis
        xtick = [str(i) if i % 2 == 1 else "" for i in range(0, 32)]
        plt.xticks(range(32), [])
        plt.xlabel("")
        # Show legend in the first plot
        plt.rcParams.update({'font.size': 20})
        plt.legend(loc='upper left')

        plt.subplot(2, 1, 2)
        kv = "v"

        draw_attn(attn_path, kv_num=10, query_index=-1, gold_index=5, label="logical", relative=relative, kv=kv)

    elif task_type == "multi_match":

        attn_path = ".//probing数据集/一个v对应3个key_vrange0-9_key在30-50-70_100kv/attn.npy"

        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        kv = "k"
        draw_attn(attn_path, kv_num=100, query_index=-36, gold_index=30, label="1st to 1st", relative=relative, kv=kv)
        draw_attn(attn_path, kv_num=100, query_index=-24, gold_index=50, label="2nd to 2nd", relative=relative, kv=kv)
        draw_attn(attn_path, kv_num=100, query_index=-12, gold_index=70, label="3rd to 3rd", relative=relative, kv=kv)
        draw_attn(attn_path, kv_num=100, query_index=-36, gold_index=50, label="1st to 2nd", relative=relative, kv=kv)
        draw_attn(attn_path, kv_num=100, query_index=-36, gold_index=70, label="1st to 3rd", relative=relative, kv=kv)

        # Hide x-axis
        xtick = [str(i) if i % 2 == 1 else "" for i in range(0, 32)]
        plt.xticks(range(32), [])
        plt.xlabel("")
        # Show legend in the first plot
        plt.rcParams.update({'font.size': 20})
        plt.legend(loc='upper left')

        plt.subplot(2, 1, 2)
        kv = "v"
        draw_attn(attn_path, kv_num=100, query_index=-36, gold_index=30, label="1st to 1st", relative=relative, kv=kv)
        draw_attn(attn_path, kv_num=100, query_index=-24, gold_index=50, label="2nd to 2nd", relative=relative, kv=kv)
        draw_attn(attn_path, kv_num=100, query_index=-12, gold_index=70, label="3rd to 3rd", relative=relative, kv=kv)
        draw_attn(attn_path, kv_num=100, query_index=-36, gold_index=50, label="1st to 2nd", relative=relative, kv=kv)
        draw_attn(attn_path, kv_num=100, query_index=-36, gold_index=70, label="1st to 3rd", relative=relative, kv=kv)

    # Hide x-axis
    xtick = [str(i) if i % 2 == 1 else "" for i in range(0, 32)]
    plt.xticks(range(32), [])
    plt.xlabel("")

    plt.rcParams.update({'font.size': 20})
    plt.legend(loc='upper left')

    # Get the current y-axis maximum value
    y_max = plt.gca().get_ylim()[1]
    plt.yticks(range(0, int(y_max), 2), fontsize=20)

    plt.xticks(range(32), xtick)

    plt.subplots_adjust(wspace=0.4, hspace=0.01)
    plt.tight_layout()
    save_path = "./对{}注意力_{}_3match_合并.png".format(kv, "rel" if relative else "abs")
    pathlib.Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.show()