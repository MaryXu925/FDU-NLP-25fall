# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn

#concat_bert_indices 不是“原始句子本身”，而是 “句子 + aspect 拼成一个 BERT 句对之后，对应的 token id 序列”。

class BERT_SPC(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1] #BERT 的输入 token ids，aspect
        attention_mask = text_bert_indices.ne(0).long() #当前位置不为 0（即不是 padding）的地方为 True。
        try:
            sequence_output, pooled_output = self.bert(
                text_bert_indices,
                token_type_ids=bert_segments_ids,
                attention_mask=attention_mask,
                return_dict=False,
            ) #pooled_output：对应 [CLS] 的 pooled 向量
        except TypeError:  # transformers<3 does not support return_dict flag
            sequence_output, pooled_output = self.bert(
                text_bert_indices,
                token_type_ids=bert_segments_ids,
                attention_mask=attention_mask,
            )
        pooled_output = self.dropout(pooled_output) #把 BERT 的 pooled_output（相当于句子级别的表示）送进 dropout 层。
        logits = self.dense(pooled_output)
        return logits
