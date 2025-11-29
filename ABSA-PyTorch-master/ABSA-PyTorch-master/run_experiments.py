# -*- coding: utf-8 -*-
"""Batch runner for ABSA experiments.

This script executes multiple model/dataset combinations, stores logs, metrics,
checkpoints, and produces simple matplotlib visualizations for each run.
"""

import argparse
import json
import os
from time import localtime, strftime

import torch

import train

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


DEFAULT_MODELS = ['bert_spc', 'lcf_bert']
DEFAULT_DATASETS = ['twitter', 'restaurant', 'laptop']


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run multiple ABSA experiments')
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                        choices=list(train.MODEL_CLASSES.keys()),
                        help='model names to evaluate')
    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS,
                        choices=list(train.DATASET_FILES.keys()),
                        help='datasets to evaluate')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--seed', type=int, default=1234, help='random seed for all runs')
    parser.add_argument('--output-dir', type=str, default='experiment_outputs',
                        help='base directory for logs, checkpoints, metrics, and figures')
    parser.add_argument('--device', type=str, default=None, help='force device, e.g. cpu or cuda:0')
    return parser.parse_args()


def prepare_opt(model_name, dataset_name, args):
    base_opt = train.build_arg_parser().parse_args([])
    base_opt.model_name = model_name
    base_opt.dataset = dataset_name
    base_opt.num_epoch = args.epochs
    base_opt.seed = args.seed
    if args.device is not None:
        base_opt.device = args.device

    base_opt.log_dir = os.path.join(args.output_dir, 'logs', model_name, dataset_name)
    base_opt.checkpoint_dir = os.path.join(args.output_dir, 'checkpoints', model_name, dataset_name)

    base_opt.model_class = train.MODEL_CLASSES[base_opt.model_name]
    base_opt.dataset_file = train.DATASET_FILES[base_opt.dataset]
    base_opt.inputs_cols = train.INPUT_COLSES[base_opt.model_name]
    base_opt.initializer = train.INITIALIZERS[base_opt.initializer]
    base_opt.optimizer = train.OPTIMIZERS[base_opt.optimizer]
    resolved_device = base_opt.device
    if resolved_device is None:
        resolved_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        resolved_device = torch.device(resolved_device)
    base_opt.device = resolved_device

    return base_opt


def save_history_plot(history, figure_path):
    if plt is None:
        train.logger.warning('matplotlib is not available; skipped plot for %s', figure_path)
        return
    if not history:
        train.logger.warning('No history to plot for %s', figure_path)
        return

    epochs = [entry['epoch'] + 1 for entry in history]
    train_acc = [entry['train_acc'] for entry in history]
    val_acc = [entry['val_acc'] for entry in history]
    train_loss = [entry['train_loss'] for entry in history]
    val_f1 = [entry['val_f1'] for entry in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, train_acc, label='train_acc')
    axes[0].plot(epochs, val_acc, label='val_acc')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('accuracy')
    axes[0].set_title('Accuracy')
    axes[0].set_xticks(epochs)
    axes[0].legend()

    axes[1].plot(epochs, train_loss, label='train_loss', color='tab:blue')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('train_loss', color='tab:blue')
    axes[1].set_title('Loss / F1')
    axes[1].set_xticks(epochs)
    axes[1].tick_params(axis='y', labelcolor='tab:blue')

    twin = axes[1].twinx()
    twin.plot(epochs, val_f1, label='val_f1', color='tab:orange')
    twin.set_ylabel('val_f1', color='tab:orange')
    twin.tick_params(axis='y', labelcolor='tab:orange')

    handles, labels = axes[1].get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    axes[1].legend(handles + twin_handles, labels + twin_labels, loc='best')

    fig.tight_layout()
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    fig.savefig(figure_path)
    plt.close(fig)


def run_experiments():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    summary_records = []

    for model_name in args.models:
        for dataset_name in args.datasets:
            train.logger.info('Starting run: model=%s dataset=%s', model_name, dataset_name)
            opt = prepare_opt(model_name, dataset_name, args)

            timestamp = strftime('%y%m%d-%H%M%S', localtime())
            log_filename = f'{model_name}-{dataset_name}-{timestamp}.log'
            os.makedirs(opt.log_dir, exist_ok=True)
            os.makedirs(opt.checkpoint_dir, exist_ok=True)
            train.setup_log_file(opt, log_filename)

            train.set_seed(opt.seed)
            instructor = train.Instructor(opt)
            result = instructor.run()

            metrics_dir = os.path.join(args.output_dir, 'metrics', model_name)
            figures_dir = os.path.join(args.output_dir, 'figures', model_name)
            os.makedirs(metrics_dir, exist_ok=True)
            os.makedirs(figures_dir, exist_ok=True)

            metrics_path = os.path.join(metrics_dir, f'{dataset_name}-{timestamp}.json')
            history = result.get('history', [])
            record = {
                'model': model_name,
                'dataset': dataset_name,
                'timestamp': timestamp,
                'best_model_path': result.get('best_model_path'),
                'test_acc': result.get('test_acc'),
                'test_f1': result.get('test_f1'),
                'log_path': getattr(opt, 'log_file', None),
                'history': history,
            }
            with open(metrics_path, 'w', encoding='utf-8') as outfile:
                json.dump(record, outfile, indent=2)

            figure_path = os.path.join(figures_dir, f'{dataset_name}-{timestamp}.png')
            save_history_plot(history, figure_path)

            summary_records.append({
                'model': model_name,
                'dataset': dataset_name,
                'timestamp': timestamp,
                'test_acc': result.get('test_acc'),
                'test_f1': result.get('test_f1'),
                'best_model_path': result.get('best_model_path'),
                'log_path': getattr(opt, 'log_file', None),
                'metrics_path': metrics_path,
                'figure_path': figure_path if history and plt is not None else None,
            })

    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        json.dump(summary_records, summary_file, indent=2)
    train.logger.info('Finished all runs. Summary saved to %s', summary_path)


if __name__ == '__main__':
    run_experiments()
