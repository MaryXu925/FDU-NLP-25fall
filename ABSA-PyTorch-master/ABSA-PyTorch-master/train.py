# -*- coding: utf-8 -*-
# file: train.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import logging
import argparse
import math
import os
import sys
import random
import numpy

from sklearn import metrics
from time import strftime, localtime

from transformers import BertModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset
from models import LSTM, IAN, MemNet, RAM, TD_LSTM, TC_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, ASGCN, LCF_BERT
from models.aen import CrossEntropyLoss_LSR, AEN_BERT
from models.bert_spc import BERT_SPC

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


MODEL_CLASSES = {
    'lstm': LSTM,
    'td_lstm': TD_LSTM,
    'tc_lstm': TC_LSTM,
    'atae_lstm': ATAE_LSTM,
    'ian': IAN,
    'memnet': MemNet,
    'ram': RAM,
    'cabasc': Cabasc,
    'tnet_lf': TNet_LF,
    'aoa': AOA,
    'mgan': MGAN,
    'asgcn': ASGCN,
    'bert_spc': BERT_SPC,
    'aen_bert': AEN_BERT,
    'lcf_bert': LCF_BERT,
}

DATASET_FILES = {
    'twitter': {
        'train': '/home/xumx/NLP/ABSA-PyTorch-master/ABSA-PyTorch-master/datasets/acl-14-short-data/train.raw',
        'test': '/home/xumx/NLP/ABSA-PyTorch-master/ABSA-PyTorch-master/datasets/acl-14-short-data/test.raw'
    },
    'restaurant': {
        'train': '/home/xumx/NLP/ABSA-PyTorch-master/ABSA-PyTorch-master/datasets/semeval14/Restaurants_Train.xml.seg',
        'test': '/home/xumx/NLP/ABSA-PyTorch-master/ABSA-PyTorch-master/datasets/semeval14/Restaurants_Test_Gold.xml.seg'
    },
    'laptop': {
        'train': '/home/xumx/NLP/ABSA-PyTorch-master/ABSA-PyTorch-master/datasets/semeval14/Laptops_Train.xml.seg',
        'test': '/home/xumx/NLP/ABSA-PyTorch-master/ABSA-PyTorch-master/datasets/semeval14/Laptops_Test_Gold.xml.seg'
    }
}

INPUT_COLSES = {
    'lstm': ['text_indices'],
    'td_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices'],
    'tc_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices', 'aspect_indices'],
    'atae_lstm': ['text_indices', 'aspect_indices'],
    'ian': ['text_indices', 'aspect_indices'],
    'memnet': ['context_indices', 'aspect_indices'],
    'ram': ['text_indices', 'aspect_indices', 'left_indices'],
    'cabasc': ['text_indices', 'aspect_indices', 'left_with_aspect_indices', 'right_with_aspect_indices'],
    'tnet_lf': ['text_indices', 'aspect_indices', 'aspect_boundary'],
    'aoa': ['text_indices', 'aspect_indices'],
    'mgan': ['text_indices', 'aspect_indices', 'left_indices'],
    'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
    'bert_spc': ['concat_bert_indices', 'concat_segments_indices'],
    'aen_bert': ['text_bert_indices', 'aspect_bert_indices'],
    'lcf_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices'],
}

INITIALIZERS = {
    'xavier_uniform_': torch.nn.init.xavier_uniform_,
    'xavier_normal_': torch.nn.init.xavier_normal_,
    'orthogonal_': torch.nn.init.orthogonal_,
}

OPTIMIZERS = {
    'adadelta': torch.optim.Adadelta,
    'adagrad': torch.optim.Adagrad,
    'adam': torch.optim.Adam,
    'adamax': torch.optim.Adamax,
    'asgd': torch.optim.ASGD,
    'rmsprop': torch.optim.RMSprop,
    'sgd': torch.optim.SGD,
}


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_spc', type=str, choices=list(MODEL_CLASSES.keys()))
    parser.add_argument('--dataset', default='laptop', type=str, choices=list(DATASET_FILES.keys()),
                        help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str, choices=list(OPTIMIZERS.keys()))
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, choices=list(INITIALIZERS.keys()))
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=20, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=85, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float,
                        help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--local_context_focus', default='cdm', type=str,
                        help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int,
                        help='semantic-relative-distance, see the paper of LCF-BERT model')
    parser.add_argument('--log_dir', default='.', type=str, help='directory to store log files')
    parser.add_argument('--checkpoint_dir', default='state_dict', type=str,
                        help='directory to store model checkpoints')
    return parser


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_log_file(opt, filename=None):
    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir, exist_ok=True)
    if filename is None:
        filename = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    log_path = os.path.join(opt.log_dir, filename)
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()
    file_handler = logging.FileHandler(log_path)
    logger.addHandler(file_handler)
    opt.log_file = log_path
    return log_path


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        self.history = []

        if 'bert' in opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer)
        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0
        path = None
        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = batch['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            train_acc = n_correct / n_total if n_total else 0
            train_loss = loss_total / n_total if n_total else 0
            logger.info('> train_acc: {:.4f}, train_loss: {:.4f}'.format(train_acc, train_loss))

            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))

            history_entry = {
                'epoch': i_epoch,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
            }
            if optimizer.param_groups:
                history_entry['lr'] = optimizer.param_groups[0].get('lr', self.opt.lr)
            self.history.append(history_entry)

            if val_acc >= max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = i_epoch
                if not os.path.exists(self.opt.checkpoint_dir):
                    os.makedirs(self.opt.checkpoint_dir, exist_ok=True)
                path = os.path.join(
                    self.opt.checkpoint_dir,
                    '{0}_{1}_val_acc_{2}'.format(self.opt.model_name, self.opt.dataset, round(val_acc, 4))
                )
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            if val_f1 >= max_val_f1:
                max_val_f1 = val_f1
            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop.')
                break

        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_batch['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return acc, f1

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        if best_model_path:
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.opt.device))
        else:
            logger.warning('No checkpoint saved during training; evaluation will use the latest model parameters.')
        test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))
        return {
            'best_model_path': best_model_path,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'history': list(self.history),
        }


def main():
    # Hyper Parameters
    parser = build_arg_parser()
    opt = parser.parse_args()

    opt.model_class = MODEL_CLASSES[opt.model_name]
    opt.dataset_file = DATASET_FILES[opt.dataset]
    opt.inputs_cols = INPUT_COLSES[opt.model_name]
    opt.initializer = INITIALIZERS[opt.initializer]
    opt.optimizer = OPTIMIZERS[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    set_seed(opt.seed)
    setup_log_file(opt)

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
