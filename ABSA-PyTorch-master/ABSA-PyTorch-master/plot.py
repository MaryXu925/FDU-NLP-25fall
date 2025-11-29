import os
import re
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_LOG_DIRS = {
    "bert_spc": {
        "twitter": os.path.join(BASE_DIR, "experiment_outputs/logs/bert_spc/twitter"),
        "restaurant": os.path.join(BASE_DIR, "experiment_outputs/logs/bert_spc/restaurant"),
        "laptop": os.path.join(BASE_DIR, "experiment_outputs/logs/bert_spc/laptop"),
    },
    "lcf_bert": {
        "twitter": os.path.join(BASE_DIR, "experiment_outputs/logs/lcf_bert/twitter"),
        "restaurant": os.path.join(BASE_DIR, "experiment_outputs/logs/lcf_bert/restaurant"),
        "laptop": os.path.join(BASE_DIR, "experiment_outputs/logs/lcf_bert/laptop"),
    },
}

COLORS = {
    "twitter": (233/255, 69/255, 49/255),        # RGB(233,69,49)
    "restaurant": (52/255, 104/255, 154/255),   # RGB(52,104,154)
    "laptop": (174/255, 174/255, 174/255),      # RGB(174,174,174)
}


def find_latest_log(log_dir):
    files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
    if not files:
        return None
    files.sort()
    return os.path.join(log_dir, files[-1])


def parse_log(path):
    epochs = []
    train_losses = []
    val_f1s = []
    current_epoch = None

    epoch_re = re.compile(r"^epoch:\s+(\d+)")
    train_re = re.compile(r"^> train_acc:\s+[\d.]+,\s+train_loss:\s+([\d.]+)")
    val_re = re.compile(r"^> val_acc:\s+[\d.]+,\s+val_f1:\s+([\d.]+)")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            m_epoch = epoch_re.match(line)
            if m_epoch:
                current_epoch = int(m_epoch.group(1))
                continue

            m_train = train_re.match(line)
            if m_train and current_epoch is not None:
                loss = float(m_train.group(1))
                epochs.append(current_epoch)
                train_losses.append(loss)
                continue

            m_val = val_re.match(line)
            if m_val and current_epoch is not None:
                f1 = float(m_val.group(1))
                val_f1s.append(f1)
                continue

    min_len = min(len(epochs), len(train_losses), len(val_f1s))
    return epochs[:min_len], train_losses[:min_len], val_f1s[:min_len]


def parse_test_f1(path):
    """从 log 的结尾解析出 test_acc 和 test_f1，如果存在的话。"""
    test_re = re.compile(r"^>> test_acc: ([\d.]+), test_f1: ([\d.]+)")
    test_acc, test_f1 = None, None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = test_re.match(line.strip())
            if m:
                test_acc = float(m.group(1))
                test_f1 = float(m.group(2))
    return test_acc, test_f1


def collect_stats(model_name: str):
    """收集某个模型在三个数据集上的 epoch / train_loss / val_f1"""
    log_dirs = MODEL_LOG_DIRS[model_name]
    datasets = ["twitter", "restaurant", "laptop"]

    all_epochs = {}
    all_train_losses = {}
    all_val_f1s = {}

    for dataset in datasets:
        log_dir = log_dirs[dataset]
        log_path = find_latest_log(log_dir)
        if log_path is None:
            print(f"[WARN] No log file found for {model_name} / {dataset} in {log_dir}")
            continue
        print(f"[INFO] Using log for {model_name} / {dataset}: {log_path}")
        epochs, train_losses, val_f1s = parse_log(log_path)
        all_epochs[dataset] = epochs
        all_train_losses[dataset] = train_losses
        all_val_f1s[dataset] = val_f1s

    return all_epochs, all_train_losses, all_val_f1s


def collect_test_f1(model_name: str):
    """收集某个模型在三个数据集上的 test_f1。"""
    log_dirs = MODEL_LOG_DIRS[model_name]
    datasets = ["twitter", "restaurant", "laptop"]
    test_f1 = {}
    for dataset in datasets:
        log_dir = log_dirs[dataset]
        log_path = find_latest_log(log_dir)
        if log_path is None:
            continue
        _, f1 = parse_test_f1(log_path)
        if f1 is not None:
            test_f1[dataset] = f1
    return test_f1


def main():
    # 先收集两个模型的统计
    spc_epochs, spc_losses, spc_f1s = collect_stats("bert_spc")
    lcf_epochs, lcf_losses, lcf_f1s = collect_stats("lcf_bert")
    spc_test_f1 = collect_test_f1("bert_spc")
    lcf_test_f1 = collect_test_f1("lcf_bert")
    datasets = ["twitter", "restaurant", "laptop"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False)
    ax_spc_loss, ax_spc_f1, ax_lcf_loss, ax_lcf_f1 = \
        axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # 1. BERT-SPC Train Loss（左上）
    for dataset in datasets:
        if dataset not in spc_epochs:
            continue
        ax_spc_loss.plot(
            spc_epochs[dataset],
            spc_losses[dataset],
            label=dataset,
            color=COLORS[dataset],
        )
    ax_spc_loss.set_title("BERT-SPC Train Loss")
    ax_spc_loss.set_xlabel("Epoch")
    ax_spc_loss.set_ylabel("Train Loss")
    ax_spc_loss.grid(True, alpha=0.3)
    ax_spc_loss.legend(fontsize=8)

    # 2. BERT-SPC Val F1（右上）
    for dataset in datasets:
        if dataset not in spc_epochs:
            continue
        ax_spc_f1.plot(
            spc_epochs[dataset],
            spc_f1s[dataset],
            label=dataset,
            color=COLORS[dataset],
        )
    ax_spc_f1.set_title("BERT-SPC Validation F1")
    ax_spc_f1.set_xlabel("Epoch")
    ax_spc_f1.set_ylabel("Val F1")
    ax_spc_f1.grid(True, alpha=0.3)
    ax_spc_f1.legend(fontsize=8)

    # 3. LCF-BERT Train Loss（左下）
    for dataset in datasets:
        if dataset not in lcf_epochs:
            continue
        ax_lcf_loss.plot(
            lcf_epochs[dataset],
            lcf_losses[dataset],
            label=dataset,
            color=COLORS[dataset],
        )
    ax_lcf_loss.set_title("LCF-BERT Train Loss")
    ax_lcf_loss.set_xlabel("Epoch")
    ax_lcf_loss.set_ylabel("Train Loss")
    ax_lcf_loss.grid(True, alpha=0.3)
    ax_lcf_loss.legend(fontsize=8)

    # 4. LCF-BERT Val F1（右下）
    for dataset in datasets:
        if dataset not in lcf_epochs:
            continue
        ax_lcf_f1.plot(
            lcf_epochs[dataset],
            lcf_f1s[dataset],
            label=dataset,
            color=COLORS[dataset],
        )
    ax_lcf_f1.set_title("LCF-BERT Validation F1")
    ax_lcf_f1.set_xlabel("Epoch")
    ax_lcf_f1.set_ylabel("Val F1")
    ax_lcf_f1.grid(True, alpha=0.3)
    ax_lcf_f1.legend(fontsize=8)

    plt.tight_layout()
    out_path = "bert_spc_lcf_4subplots.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved figure: {out_path}")

    # 额外画两张图：每个模型在三个数据集上的 test_f1
    # test F1 图中从左到右的顺序：restaurant, laptop, twitter
    test_datasets = ["restaurant", "laptop", "twitter"]

    plt.figure(figsize=(5, 4))
    x = range(len(test_datasets))
    # 转换为百分制（0-100）
    y = [spc_test_f1.get(ds, 0.0) * 100 for ds in test_datasets]
    plt.bar(x, y, color=[COLORS[d] for d in test_datasets])
    plt.xticks(x, test_datasets)
    # 在每个柱子上标出具体数值
    for i, v in enumerate(y):
        plt.text(i, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    plt.ylim(50, 90)
    plt.ylabel("Test F1 (%)")
    plt.title("BERT-SPC Test F1")
    plt.tight_layout()
    spc_out = "bert_spc_test_f1.png"
    plt.savefig(spc_out, dpi=200)

    plt.figure(figsize=(5, 4))
    # 转换为百分制（0-100）
    y = [lcf_test_f1.get(ds, 0.0) * 100 for ds in test_datasets]
    plt.bar(x, y, color=[COLORS[d] for d in test_datasets])
    plt.xticks(x, test_datasets)
    # 在每个柱子上标出具体数值
    for i, v in enumerate(y):
        plt.text(i, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    plt.ylim(50, 90)
    plt.ylabel("Test F1 (%)")
    plt.title("LCF-BERT Test F1")
    plt.tight_layout()
    lcf_out = "lcf_bert_test_f1.png"
    plt.savefig(lcf_out, dpi=200)

    print(f"Saved test F1 figures: {spc_out}, {lcf_out}")


if __name__ == "__main__":
    main()