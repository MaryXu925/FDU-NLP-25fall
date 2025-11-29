# transformer.py

import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *


class CausalSelfAttention(nn.Module):
    # 带因果掩码的注意力子层
    def __init__(self, config): #接收一个 config 对象（应包含 n_embd 和 n_head 等超参数）
        super().__init__()
        assert config.n_embd % config.n_head == 0 
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization / config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.causal = getattr(config, 'causal', True)

    def forward(self, x, return_weights: bool=False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # Project and split QKV
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # reshape to heads
        head_dim = C // self.n_head
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2) # (B, nh, T, hs)

        # manual scaled dot-product attention to optionally expose weights
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim) # (B, nh, T, T)
        if self.causal:
            causal_mask = torch.triu(torch.ones(T, T, device=scores.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1) # (B, nh, T, T)
        y = torch.matmul(attn, v) # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)
        # output projection
        y = self.c_proj(y)
        if return_weights:
            return y, attn
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, n_head: int = 4, causal: bool = True):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.num_positions = num_positions
        self.d_model = d_model
        self.d_internal = d_internal
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.n_head = n_head
        self.causal = causal

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional = PositionalEncoding(d_model, num_positions=num_positions, batched=False)
        self.layers = nn.ModuleList([TransformerLayer(d_model, d_internal, n_head=self.n_head, causal=self.causal) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        if indices.dim() != 1:
            indices = indices.view(-1)
        x = self.embedding(indices)  # [T, d_model]
        x = self.positional(x)       # [T, d_model]
        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x)
            attn_maps.append(attn)
        x = self.ln_f(x)
        logits = self.classifier(x)  # [T, num_classes]
        log_probs = F.log_softmax(logits, dim=1)
        return log_probs, attn_maps


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal, n_head: int = 1, causal: bool = True):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_internal

        # Pre-LN Transformer block
        self.ln_1 = nn.LayerNorm(d_model)
        # reuse the provided CausalSelfAttention (multi-head supported)
        class _Cfg:
            def __init__(self, n_embd, n_head, causal):
                self.n_embd = n_embd
                self.n_head = n_head
                self.causal = causal
        self.self_attn = CausalSelfAttention(_Cfg(d_model, n_head, causal))

        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout_ff = nn.Dropout(0.1)

    def forward(self, input_vecs):
        """
        :param input_vecs: an input tensor of shape [seq len, d_model]
        :return: a tuple of two elements:
            - a tensor of shape [seq len, d_model] representing the log probabilities of each position in the input
            - a tensor of shape [seq len, seq len], representing the attention map for this layer
        """
        x = input_vecs  # [T, d_model]
        # Self-attention using CausalSelfAttention (return attention weights)
        x_norm = self.ln_1(x)
        y, attn = self.self_attn(x_norm.unsqueeze(0), return_weights=True)  # y: [1,T,C], attn: [1,nh,T,T]
        x = x + y.squeeze(0)

        # Average heads to get [T, T]
        attn_map = attn.mean(dim=1).squeeze(0)

        # Feed-forward
        y = self.ln_2(x)
        y = self.ffn(y)
        y = self.dropout_ff(y)
        x = x + y

        return x, attn_map


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module): # 可学习位置编码
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    # Hyperparameters
    vocab_size = 27
    num_positions = 20
    d_model = 128
    d_internal = 128
    num_classes = 3
    num_layers = 2
    n_head = 4
    causal = True if getattr(args, 'task', 'BEFORE') == 'BEFORE' else False

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, n_head=n_head, causal=causal)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fcn = nn.NLLLoss()

    num_epochs = 10
    for t in range(num_epochs):
        loss_this_epoch = 0.0
        ex_idxs = list(range(len(train)))
        random.shuffle(ex_idxs)
        for idx in ex_idxs:
            ex = train[idx]
            log_probs, _ = model(ex.input_tensor)
            loss = loss_fcn(log_probs, ex.output_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
        print(f"Epoch {t+1}/{num_epochs} - Loss: {loss_this_epoch/len(train):.6f}")

    # Ensure plots directory exists for decode() attention map saving
    try:
        os.makedirs("plots", exist_ok=True)
    except Exception:
        pass

    model.eval()
    return model

    # The following code DOES NOT WORK but can be a starting point for your implementation
    # Some suggested snippets to use:
    model = Transformer(...)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        random.seed(t)
        # You can use batching if you'd like
        ex_idxs = [i for i in range(0, len(train))]
        random.shuffle(ex_idxs)
        loss_fcn = nn.NLLLoss()
        for ex_idx in ex_idxs:
            loss = loss_fcn(...) # TODO: Run forward and compute loss
            # model.zero_grad()
            # loss.backward()
            # optimizer.step()
            loss_this_epoch += loss.item()
    model.eval()
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
