import os
import yaml
import json
import pickle
import random
import torch
import matplotlib.pyplot as plt
import numpy as np

from pynvml import *
from IPython import display
from contextlib import contextmanager
from matplotlib.lines import Line2D
from sklearn.metrics import f1_score, precision_score, recall_score
from torchvision.transforms import functional as TF
from torchmetrics.classification import (
    MulticlassAccuracy,
    MultilabelAccuracy,
    MulticlassF1Score,
    MultilabelF1Score,
    MulticlassPrecision,
    MultilabelPrecision,
    MulticlassRecall,
    MultilabelRecall,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # safe to call even when the GPU is not available

    # When running on the CuDNN backend, two further options must be set
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}.")


# Util for saving objects in pickle format.
def save_pickle(file_path, obj):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


# Util for loading objects from pickle format.
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def print_separator(text='', seperator_length=20, seperator='='):
    print('\n', seperator*seperator_length + ' ' + text + ' ' + seperator*seperator_length + '\n')


# Util for printing result at the end of each round.
def print_result(performance_log):
    print('train     -              loss :    {:.4f}          acc:    {:.4f}'.format(performance_log['train_loss'][-1], performance_log['train_acc'][-1]))
    print('valid     -              loss :    {:.4f}          acc:    {:.4f}'.format(performance_log['valid_loss'][-1], performance_log['valid_acc'][-1]))
    print()


# For plotting model history.
def save_history_plot(history_log, plot_config):
    
    plt.figure()
    for attribute, label in zip(plot_config['attributes'], plot_config['labels']):
        plt.plot(history_log[attribute], label=label)
    plt.title(plot_config['title'])
    plt.xlabel(plot_config['xlabel'])
    plt.ylabel(plot_config['ylabel'])
    plt.grid(True, linestyle='-.')
    plt.legend()

    if plot_config['save_dir'] != None:
        plt.savefig(plot_config['save_dir'], dpi=200, bbox_inches='tight')
    if plot_config['show_img']:
        plt.show()

    plt.close('all')


def timeformat(total_seconds):

    total_seconds = int(total_seconds)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = (total_seconds % 3600) % 60
    
    formatted_time = str(hours).zfill(2) + ':' + str(minutes).zfill(2) + ':' + str(seconds).zfill(2)
    return formatted_time


# Helper for computing metrics at each epoch.
class MeanMetric():
    
    def __init__(self):
        self.total = np.float32(0)
        self.count = np.float32(0)

    def update_state(self, value):
        self.total += value
        self.count += 1
        
    def result(self):
        if self.count > 0:
            return self.total / self.count
        else:
            return np.nan

    def reset_state(self):
        self.total = np.float32(0)
        self.count = np.float32(0)


# Plotting settings
LOSS_PLOT_CONFIG = {
    'figsize' : (8, 6),
    'attributes': ('train_loss', 'valid_loss'),
    'labels': ('train', 'valid'),
    'title': 'Loss',
    'xlabel': 'rounds',
    'ylabel': 'loss',
    'save_dir': None,   
    'show_img': False,
    'x_axis_data': None
}

ACC_PLOT_CONFIG = {
    'figsize' : (8, 6),
    'attributes': ['train_acc', 'valid_acc'],
    'labels': ['train', 'valid'],
    'title': 'Accuracy',
    'xlabel': 'rounds',
    'ylabel': 'accuracy',
    'save_dir': None,
    'show_img': False,
    'x_axis_data': None
}

# For plotting model history.
def save_history_plot(history_log, plot_config):
    
    plt.figure(figsize=plot_config['figsize'])
    for attribute, label in zip(plot_config['attributes'], plot_config['labels']):
        plt.plot(history_log[attribute], label=label)
    plt.title(plot_config['title'])
    plt.xlabel(plot_config['xlabel'])
    plt.ylabel(plot_config['ylabel'])
    plt.grid(True, linestyle='-.')
    plt.legend()

    if plot_config['save_dir'] != None:
        plt.savefig(plot_config['save_dir'], dpi=300, bbox_inches='tight')
    if plot_config['show_img']:
        plt.show()

    plt.close('all')


def save_plot(data_list, plot_config):
    plt.figure(figsize=plot_config['figsize'])

    if plot_config['x_axis_data'] is not None:
        for x_data, y_data, label in zip(plot_config['x_axis_data'], data_list, plot_config['labels']):
            plt.plot(x_data, y_data, label=label)
    else:
        for data, label in zip(data_list, plot_config['labels']):
            plt.plot(data, label=label)
    plt.title(plot_config['title'])
    plt.xlabel(plot_config['xlabel'])
    plt.ylabel(plot_config['ylabel'])
    plt.grid(True, linestyle='-.')
    plt.legend()

    if plot_config['save_dir'] is not None:
        plt.savefig(plot_config['save_dir'], dpi=300, bbox_inches='tight')
    if plot_config['show_img']:
        plt.show()

    plt.close('all')


# Helpers for logging model performance.
def get_performance_loggers(metric_keys = {'train_loss', 'train_acc', 'valid_loss', 'valid_acc'}):
	performance_dict, performance_log = dict(), dict()
	for key in metric_keys:
	    performance_dict[key] = MeanMetric()
	    performance_log[key] = list()
	return performance_dict, performance_log


# GPU memory allocation workaround.
def allocate_gpu_memory(mem_amount=10504699904):
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total}')
    print(f'free     : {info.free}')
    print(f'used     : {info.used}')

    block_mem = int(info.free * 0.7) // (32 // 8)
    block_mem = int(block_mem * 0.7)
    block_mem

    device_count = torch.cuda.device_count()
    for i in range(device_count):
        device = torch.device('cuda:{}'.format(i) if torch.cuda.is_available() else 'cpu')
        x = torch.rand(block_mem, dtype=torch.float32).to(device)
        x = torch.rand(1)
        del x

    nvmlShutdown()


def save_notes(file_path, notes):
    with open(file_path, 'w') as f:
        f.write(notes)


# Util for showing results.
def get_mean_std(num_list):
    print('mean:{:.2f}'.format(np.mean(num_list)))
    print('std:{:.2f}'.format(np.std(num_list)))
    

def show_img_tensor(img_tensor, save_path=None, dpi=1200):
    img_tensor = img_tensor.cpu()
    img = TF.to_pil_image(img_tensor.add(1).div(2).clamp(0, 1))
    plt.axis('off')
    plt.imshow(img)
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


# Utilities
@contextmanager
def train_mode(model, mode=True):
    """A context manager that places a model into training mode and restores
    the previous mode on exit."""
    modes = [module.training for module in model.modules()]
    try:
        yield model.train(mode)
    finally:
        for i, module in enumerate(model.modules()):
            module.training = modes[i]


def eval_mode(model):
    """A context manager that places a model into evaluation mode and restores
    the previous mode on exit."""
    return train_mode(model, False)


def save_dict_yaml(file_path, note_dict):
    with open(file_path, 'w') as yaml_file:
            yaml.dump(note_dict, yaml_file, default_flow_style=False)


def save_dict_json(file_path, note_dict):
    with open(file_path, 'w') as f:
            f.write(json.dumps(note_dict))


def plot_grad_flow_line(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        
        if(p.requires_grad) and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


def plot_grad_flow_bar(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


# Compute accuracy.
def compute_accuracy(label, label_pred, is_multilabel=False):

    device = label.device
    num_classes = label_pred.shape[-1]
    
    if not is_multilabel:
        accuracy_metric = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)  # Average needs to be micro.
    elif is_multilabel:
        accuracy_metric = MultilabelAccuracy(num_labels=num_classes, average='micro').to(device)  # Average needs to be micro.
    
    accuracy = accuracy_metric(label_pred, label)
    accuracy = accuracy.item()
    return accuracy


# Using torchmetrics.
def compute_precision_recall_f1(label, label_pred, is_multilabel=False, average='macro'):
    
    device = label.device
    num_classes = label_pred.shape[-1]

    if not is_multilabel:
        f1_metric = MulticlassF1Score(num_classes=num_classes, average=average).to(device)
        precision_metric = MulticlassPrecision(num_classes=num_classes, average=average).to(device)
        recall_metric = MulticlassRecall(num_classes=num_classes, average=average).to(device)
    elif is_multilabel:
        f1_metric = MultilabelF1Score(num_labels=num_classes, average=average).to(device)
        precision_metric = MultilabelPrecision(num_labels=num_classes, average=average).to(device)
        recall_metric = MultilabelRecall(num_labels=num_classes, average=average).to(device)

    f1 = f1_metric(label_pred, label)
    f1 = f1.item()

    precision = precision_metric(label_pred, label)
    precision = precision.item()

    recall = recall_metric(label_pred, label)
    recall = recall.item()

    return precision, recall, f1