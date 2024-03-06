import os
import numpy as np
import glob
# try:
#     import cv2
# except ImportError as e:
#     print(e)
import cv2
import re
import sys
import random
import math
import copy
import time
import datetime
import json
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from random import shuffle
from itertools import islice,chain
from pprint import pprint
import seaborn as sns


import torch
print("TORCH VERSION: "+torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
# PyTorchで画像認識に使用するネットワークやデータセットを利用するためのモジュール
import torchvision
# PyTorchでメトリクスを算出するためのモジュール
import torchmetrics
# PyTorchでネットワーク構造を確認するためのモジュール
import torchsummary


from common_func import *


GPU_COUNT = torch.cuda.device_count()
if torch.cuda.is_available():
    print("ALL GPU COUNT: "+str(torch.cuda.device_count()))
    print("ALL GPU NAMES: "+str([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))
print()









device = torch.device("cuda")



def train(args, logging, train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, scheduler, start_step=0, metrics_dict={}):
    objs = AvgrageMeter()
    obja = AvgrageMeter()

    if hasattr(args, 'warmup') and args.warmup and epoch < args.warmup_epoch:
        logging.info('epoch %d warming up!!!', epoch)
    else:
        logging.info('epoch %d train arch!!!', epoch)

    # metricsからROCは削除
    if 'roc' in metrics_dict:
        metrics_dict.pop('roc')

    # metrics
    metrics = {}
    for k, _ in metrics_dict.items():
        metrics[k] = []

    history_epoch = {}
    t_sum = 0
    t_sum_part = 0
    bar_length = 20
    checkpoint_interval = math.floor(len(train_queue) / 30)
    for step, (input, target) in enumerate(train_queue,start=1):
        if step < start_step+1:
            continue
        t = time.time()
        model.train()
        n = input.size(0)

        input = input.to(device)
        target = target.to(device)

        if hasattr(args, 'warmup') and args.warmup and epoch < args.warmup_epoch:
            pass
        else:
            # get a random minibatch from the search queue with replacement
            input_search, target_search = next(iter(valid_queue))
            #try:
            #  input_search, target_search = next(valid_queue_iter)
            #except:
            #  valid_queue_iter = iter(valid_queue)
            #  input_search, target_search = next(valid_queue_iter)
            input_search = input_search.to(device)
            target_search = target_search.to(device)

            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        # その他のMetricsの計算
        for k, fn in metrics_dict.items():
            metric = fn(logits,target.long())
            metrics[k].append(metric.item())

        loss.backward()
        nn.utils.clip_grad_norm(model.module.parameters(), args.grad_clip)
        optimizer.step()

        acc = accuracy_binary(logits, target)
        objs.update(loss.item(), n)
        obja.update(acc.item(), n)

        # if step % args.report_freq == 0:
        #   logging.info('train %03d %e %f', step, objs.avg, obja.avg)

        if step % checkpoint_interval == 0:
            past_time = load_checkpoint_past_time(os.path.join(args.save, 'state.pt'))
            if past_time is None:
                past_time = 0
            past_time = float(past_time)
            state = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'architect_optim': architect.optimizer.state_dict(),
                'epoch': epoch-1,
                'step': step,
                'loss': objs.avg,
                'train_elapsed_time': past_time+t_sum_part
            }
            save_checkpoint(state, os.path.join(args.save, 'state.pt'))
            t_sum_part = 0

        interval = time.time() - t
        t_sum += interval
        t_sum_part += interval
        eta = str(datetime.timedelta( seconds=int((len(train_queue)-step)*interval) ))
        done = math.floor(bar_length * step / len(train_queue))
        bar = '='*done + ("=" if done == bar_length else ">")  + "."*(bar_length-done)
        print_log(f"\r  \033[K[{bar}] - ETA: {eta:>8}, {math.floor(step / len(train_queue)*100):>3}% ({step}/{len(train_queue)})", line_break=False)
        print_log(f", ElapsedTime: {interval:.07f}", line_break=False)
        print_log(f", Loss: {objs.avg:.04f}", line_break=False)
        print_log(f", Acc: {obja.avg:.04f}", line_break=False)
        print_log(f"\t", line_break=False)
    t_total = str(datetime.timedelta( seconds=int(t_sum) ))
    bar = '='*bar_length
    history_epoch['elapsed_time'] = t_total
    history_epoch['loss'] = objs.avg
    print_log(f"\r  \033[K[{bar}] - {t_total:>8}, 100% ({len(train_queue)}/{len(train_queue)})", line_break=False)
    print_log(f", Loss: {objs.avg:.04f}", line_break=False)
    print_log(f", Acc: {obja.avg:.04f}", line_break=False)
    for k,v in metrics.items():
        if len(v)==0:
            history_epoch[k] = 0
            continue
        train_metric = sum(v) / len(v)
        exec('{} = {}'.format(k, train_metric))
        print_log(f", {k.capitalize()}: {train_metric:.04f}", line_break=False)
        history_epoch[k] = train_metric
    print_log(f"\t", line_break=False)
    print_log()

    return obja.avg, objs.avg, history_epoch


def infer(args, logging, valid_queue, model, criterion, metrics_dict={}):
    objs = AvgrageMeter()
    obja = AvgrageMeter()
    model.eval()

    # metrics
    metrics = {}
    for k, _ in metrics_dict.items():
        metrics[k] = []

    history = {}
    bar_length = 20
    t_test = 0
    batch_num = len(valid_queue)

    with torch.no_grad():
        batch_start_time = time.time()
        for step, (input, target) in enumerate(valid_queue):
            #input = input.cuda()
            #target = target.cuda(non_blocking=True)
            input = input.to(device)
            target = target.to(device)
            logits = model(input)
            loss = criterion(logits, target)

            # その他のMetricsの計算
            for k, fn in metrics_dict.items():
                if k=='roc':
                    fn.update(logits,target.long())
                else:
                    metric = fn(logits,target.long())
                    metrics[k].append(metric.item())

            acc = accuracy_binary(logits, target)
            n = input.size(0)
            objs.update(loss.item(), n)
            obja.update(acc.item(), n)

            # if step % args.report_freq == 0:
            #   logging.info('valid %03d %e %f', step, objs.avg, obja.avg)

            interval = time.time() - batch_start_time
            t_test += interval
            eta = str(datetime.timedelta(seconds= int((batch_num-step+1)*interval) ))
            done = math.floor(bar_length * step / batch_num)
            bar = '='*done + ("=" if done == bar_length else ">")  + "."*(bar_length-done)
            print_log(f"\r  \033[K[{bar}] - ETA: {eta:>8}, {math.floor(step / batch_num*100):>3}% ({step}/{batch_num})", line_break=False)
            batch_start_time = time.time()

        done = bar_length
        bar = '='*done + ("=" if done == bar_length else ">")  + "."*(bar_length-done)
        print_log(f"\r  \033[K[{bar}] - {int(t_test)}s, 100% ({batch_num}/{batch_num})", line_break=False)

        history['loss'] = objs.avg
        for k,v in metrics.items():
            if k!='roc':
                test_metric = sum(v) / len(v)
                print_log(f", {k.capitalize()}: {test_metric:.04f}", line_break=False)
                history[k] = test_metric
        if 'roc' in metrics_dict:
            history['roc'] = metrics_dict['roc'].compute()
        print_log(f"\t", line_break=False)
        print_log()

    return obja.avg, objs.avg, history


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt










def accuracy(output, target):
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[0].reshape(-1).float().sum(0)
    res = correct_k.mul_(100.0/batch_size)

    return res

def accuracy_binary(output, target):
    batch_size = target.size(0)
    pred = (output > 0.5).float()
    correct = (pred == target).float().sum()
    return (correct / batch_size) * 100


def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    architect_optim = state['architect_optim']
    epoch = state['epoch']
    step = state['step']
    return model, optimizer, scheduler, architect_optim, epoch, step

def load_checkpoint_past_time(checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        return 0
    state = torch.load(checkpoint_path)
    train_elapsed_time = state['train_elapsed_time'] if 'train_elapsed_time' in state else None
    return train_elapsed_time


### ログ表示 ###
def print_log(message="", line_break=True):
    if line_break:
        sys.stdout.write(message + "\n")
    else:
        sys.stdout.write(message)
    sys.stdout.flush()

### jsonパラメータ保存 ###
def saveArgs(args, filename="./params.json"):
    params = {}
    for arg_name, arg_value in vars(args).items():
        params[arg_name] = arg_value
    with open(filename, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=False, ensure_ascii=False)

### jsonパラメータ読込 ###
def loadArgs(args, filename="./params.json"):
    with open(filename, 'r') as f:
        params = json.load(f)
    if 'train_rate' not in params:
        setattr(args, 'train_rate', None)
    if 'use_data_rate' not in params:
        setattr(args, 'use_data_rate', 1.0)
    for arg_name, arg_value in params.items():
        if arg_name in vars(args):
            setattr(args, arg_name, arg_value)
    return params


### MNISTロード用 ###
def get_indices(dataset, class_names):
    indices = []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] in class_names:
            indices.append(i)
    return indices
