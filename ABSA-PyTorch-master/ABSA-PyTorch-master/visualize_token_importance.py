# -*- coding: utf-8 -*-
# file: visualize_token_importance_lcf_multi.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from transformers import BertModel
from data_utils import Tokenizer4Bert, ABSADataset
from models.lcf_bert import LCF_BERT

# 手动设置：使用哪个 checkpoint、数据集
CHECKPOINT_PATH = "/home/xumx/NLP/ABSA-PyTorch-master/ABSA-PyTorch-master/state_dict/lcf_bert_twitter_val_acc_0.7399"
PRETRAINED_BERT_NAME = "bert-base-uncased"
MAX_SEQ_LEN = 85
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET = "twitter"
SPLIT = "test"

DATASET_FILES = {
    'twitter': {
        'train': 'datasets/acl-14-short-data/train.raw',
        'test': 'datasets/acl-14-short-data/test.raw'
    },
    'restaurant': {
        'train': 'datasets/semeval14/Restaurants_Train.xml.seg',
        'test': 'datasets/semeval14/Restaurants_Test_Gold.xml.seg'
    },
    'laptop': {
        'train': 'datasets/semeval14/Laptops_Train.xml.seg',
        'test': 'datasets/semeval14/Laptops_Test_Gold.xml.seg'
    }
}

# 复用 plot.py 里的三种颜色
CASE_COLORS = [
    (233/255, 69/255, 49/255),      # twitter color
    (52/255, 104/255, 154/255),     # restaurant color
    (174/255, 174/255, 174/255),    # laptop color
]


def build_lcf_bert_model():
    class Opt:
        pass

    opt = Opt()
    opt.dropout = 0.1
    opt.bert_dim = 768
    opt.polarities_dim = 3
    opt.max_seq_len = MAX_SEQ_LEN
    opt.local_context_focus = 'cdm'
    opt.SRD = 3
    opt.device = DEVICE

    bert = BertModel.from_pretrained(PRETRAINED_BERT_NAME)
    model = LCF_BERT(bert, opt)
    return model, opt


def load_model(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def compute_token_importance(last_hidden_state, method="cls_cosine"):
    hidden = last_hidden_state[0]  # [seq_len, hidden_size]

    if method == "cls_cosine":
        cls_vec = hidden[0]
        cls_norm = cls_vec / (cls_vec.norm(p=2) + 1e-8)
        token_norm = hidden / (hidden.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        scores = (token_norm * cls_norm).sum(dim=-1)
    elif method == "l2_norm":
        scores = hidden.norm(p=2, dim=-1)
    else:
        raise ValueError("Unknown method: {}".format(method))

    scores = scores.detach().cpu().numpy()
    scores = scores - scores.min()
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores


def extract_local_importance_for_sample(model, opt, tokenizer, sample):
    # 转 tensor
    concat_bert_indices = torch.tensor(sample['concat_bert_indices'], dtype=torch.long).unsqueeze(0).to(DEVICE)
    concat_segments_indices = torch.tensor(sample['concat_segments_indices'], dtype=torch.long).unsqueeze(0).to(DEVICE)
    text_bert_indices = torch.tensor(sample['text_bert_indices'], dtype=torch.long).unsqueeze(0).to(DEVICE)
    aspect_bert_indices = torch.tensor(sample['aspect_bert_indices'], dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        attention_mask_global = concat_bert_indices.ne(0).long()
        attention_mask_local = text_bert_indices.ne(0).long()

        # 全局 BERT（句对）
        bert_spc_out, _ = model.bert_spc(
            concat_bert_indices,
            token_type_ids=concat_segments_indices,
            attention_mask=attention_mask_global,
            return_dict=False,
        )
        bert_spc_out = model.dropout(bert_spc_out)

        # 本地 BERT（只看 text）
        bert_local_out, _ = model.bert_local(
            text_bert_indices,
            attention_mask=attention_mask_local,
            return_dict=False,
        )
        bert_local_out = model.dropout(bert_local_out)

        # LCF cdm/cdw
        if opt.local_context_focus == 'cdm':
            masked_local_text_vec = model.feature_dynamic_mask(text_bert_indices, aspect_bert_indices)
            bert_local_lcf = torch.mul(bert_local_out, masked_local_text_vec)
        elif opt.local_context_focus == 'cdw':
            weighted_text_local_features = model.feature_dynamic_weighted(text_bert_indices, aspect_bert_indices)
            bert_local_lcf = torch.mul(bert_local_out, weighted_text_local_features)
        else:
            bert_local_lcf = bert_local_out

    # 计算 local token 重要度
    local_scores = compute_token_importance(bert_local_lcf, method="cls_cosine")
    bert_tokenizer = tokenizer.tokenizer
    text_ids = text_bert_indices[0].tolist()
    if 0 in text_ids:
        pad_start = text_ids.index(0)
        text_ids = text_ids[:pad_start]
        local_scores = local_scores[:pad_start]
    local_tokens = bert_tokenizer.convert_ids_to_tokens(text_ids)

    return local_tokens, local_scores


def main():
    tokenizer = Tokenizer4Bert(MAX_SEQ_LEN, PRETRAINED_BERT_NAME)
    dataset_path = DATASET_FILES[DATASET][SPLIT]
    dataset = ABSADataset(dataset_path, tokenizer)

    # 选择 3 个 case 的索引，你可以手动改，例如 [0, 10, 20]
    sample_indices = [0, 13, 73]

    model, opt = build_lcf_bert_model()
    model = load_model(model, CHECKPOINT_PATH)

    cases = []
    for idx in sample_indices:
        sample = dataset[idx]
        tokens, scores = extract_local_importance_for_sample(model, opt, tokenizer, sample)
        cases.append((idx, tokens, scores))

    # 画三行子图，每行一个 case，各占 1/3
    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=False)

    for i, (idx, tokens, scores) in enumerate(cases):
        ax = axes[i]
        x = np.arange(len(tokens))
        color = CASE_COLORS[i % len(CASE_COLORS)]

        ax.bar(x, scores, color=color)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("importance")
        ax.set_title(f"LCF local token importance (sample idx={idx})")
        ax.set_xticks(x)
        ax.set_xticklabels(tokens, rotation=90, fontsize=8)

    plt.tight_layout()
    out_path = "lcf_twitter_local_3cases.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()