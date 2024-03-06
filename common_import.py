
import datetime
# 現在時刻取得
def now(format_str="%Y-%m-%d %H:%M:%S"):
    now = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    return now.strftime(format_str)
print(f"START IMPORT: {now()}")
print()

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
# import torch.nn as nn
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

import nni
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
print("NII VERSION: "+nni.__version__)
print()


from common_func import *


##### GPU1枚の最大メモリ量 #####
GPU_MEMORY_LIMIT = 31.0 #RAIDEN
# GPU_MEMORY_LIMIT = 11.0 #GHPC


##### デバイスを指定 #####
device = 'cuda'
# device = 'cuda:1'
# device = 'cpu'

GPU_COUNT = torch.cuda.device_count() if device == 'cuda' else 1
if torch.cuda.is_available() and (device != 'cpu'):
    print("DEVICE: "+str(device))
    print("ALL GPU COUNT: "+str(torch.cuda.device_count()))
    print("ALL GPU NAMES: "+str([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))
    if device == 'cuda':
        print("USE GPU NAMES: "+str([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))
    else:
        print("USE GPU NAMES: "+torch.cuda.get_device_name(device))
else:
    print("USE CPU")
    device = 'cpu'
device = torch.device(device)
print()


##### 環境変数読み込み #####
from dotenv import load_dotenv
load_dotenv()


















####################################################################################################
# pytorch 便利関数
####################################################################################################

### 学習 ###
def train(model, device, train_dataloader, loss_fn, optimizer, epoch, validation_dataloader=None, cp_step=None, cp_path="./model_weights_{epoch:03d}_{loss:.4f}.pth", project_dir="./", es_flg=False, es_patience=7, callbacks=None, initial_epoch=0, metrics_dict={}):
    """
    pytorchのモデルを学習する。
    
    Parameters
    ----------
    model : torch.nn.Module
        学習対象のモデル
    device : torch.cuda.device or str
        使用するデバイス
    train_dataloder : torch.utils.data.DataLoader
        学習データ
    loss_fn : torch.nn.lossFunctions
        損失関数
    optimizer : torch.optim.*
        最適化関数
    epoch : int
        エポック
    validation_dataloder : torch.utils.data.DataLoader
        検証データ
    cp_step : None or int
        何エポックごとに重みを保存するか。1の場合、毎エポック。Noneの場合は保存しない。
    cp_path : str
        重みを保存するパス
    callback : dict or None
        コールバック関数。実行したい位置によって辞書のkeyを指定する。
            on_function_begin   -> 関数開始前
            on_epoch_begin      -> エポック開始前
            on_train_begin      -> 学習開始前
            on_train_end        -> 学習終了後
            on_validation_begin -> 検証開始前
            on_validation_end   -> 検証終了後
            on_epoch_end        -> エポック終了後
            on_function_end     -> 関数終了後
    initial_epoch : int
        前回までに学習したエポック数。これまでに10エポック学習した場合、10を指定して11から学習開始。初回の学習の場合は0を指定。
    metrics_dict : dict
        計算したいMetricsの辞書を指定。getMetrics関数によって作成されたものをそのまま指定。

    Returns
    -------
    history : dict
        学習結果
    """

    # callbackの実行
    if (callbacks is not None) and ('on_function_begin' in callbacks) and callable(callbacks['on_function_begin']):
        callbacks['on_function_begin'](model=model)

    # 変数
    val_flg = True if validation_dataloader is not None else False # validation_dataがあるか
    history = {'loss':[],'training_elapsed_time':[],'epoch_elapsed_time':[]} # 返り値用変数
    if val_flg:
        history['val_loss'] = []
        history['val_elapsed_time'] = []

    # history初期化
    for k, _ in metrics_dict.items():
        history[k] = []
        if val_flg:
            history["val_"+k] = []

    # 過去のhistoryを読み込み
    past_history = None
    if os.path.isfile(project_dir+"/history.json"):
        past_history = loadParams(filename=project_dir+"/history.json")
    # print(past_history)

    # early stoppingを設定
    if es_flg:
        es = EarlyStopping(patience=es_patience, verbose=False, delta=0, path=project_dir+'/checkpoint_minloss.pth')
        if (past_history is not None) and ('loss' in past_history):
            for loss in past_history['loss']:
                es(loss,model)
                if es.early_stop:
                    print('Losses have already risen more than {es_patience} times.')
                    return

    # 学習を繰り返し行う
    for epoch_idx in range(initial_epoch,initial_epoch+epoch):
        
        epoch_start_time = time.time()
        history_epoch = {}

        # callbackの実行
        if (callbacks is not None) and ('on_epoch_begin' in callbacks) and callable(callbacks['on_epoch_begin']):
            callbacks['on_epoch_begin'](model=model)

        print_log(f"Epoch: {epoch_idx+1:>3}/{initial_epoch+epoch}")

        ###学習###
        train_history = train_one_epoch(model, device, train_dataloader, loss_fn, optimizer, callbacks, metrics_dict)
        for k, v in train_history.items():
            history[k].append(v)
            history_epoch[k] = v

        ###検証###
        if val_flg:
            validation_history = validation_one_epoch(model, device, validation_dataloader, loss_fn, callbacks, metrics_dict)
            for k, v in validation_history.items():
                history[k].append(v)
                history_epoch[k] = v 

        # モデルの重みの保存(check point)
        if (cp_step is not None) and ((epoch_idx+1) % cp_step == 0):
            torch.save(
                {
                    'epoch': epoch_idx+1,
                    'loss': train_history['loss'],
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                cp_path.format(
                    epoch=epoch_idx+1,
                    loss=train_history['loss'],
                    accuracy=train_history['accuracy'] if ('accuracy' in train_history) else None,
                    acc=train_history['accuracy'] if ('accuracy' in train_history) else None,
                    auc=train_history['auc'] if ('auc' in train_history) else None,
                    precision=train_history['precision'] if ('precision' in train_history) else None,
                    recall=train_history['recall'] if ('recall' in train_history) else None,
                    specificity=train_history['specificity'] if ('specificity' in train_history) else None,
                    f1=train_history['f1'] if ('f1' in train_history) else None,
                    val_loss=validation_history['val_loss'] if val_flg and ('val_loss' in validation_history) else None,
                    val_accuracy=validation_history['val_accuracy'] if val_flg and ('val_accuracy' in validation_history) else None,
                    val_acc=validation_history['val_accuracy'] if val_flg and ('val_accuracy' in validation_history) else None,
                    val_auc=validation_history['val_auc'] if val_flg and ('val_auc' in validation_history) else None,
                    val_precision=validation_history['val_precision'] if val_flg and ('val_precision' in validation_history) else None,
                    val_recall=validation_history['val_recall'] if val_flg and ('val_recall' in validation_history) else None,
                    val_specificity=validation_history['val_specificity'] if val_flg and ('val_specificity' in validation_history) else None,
                    val_f1=validation_history['val_f1'] if val_flg and ('val_f1' in validation_history) else None
                )
            )

        # モデルの重みの保存(最良モデル)
        if (past_history is not None) and (len(past_history['loss']) > 0) and (train_history['loss'] < min(past_history['loss'])):
            torch.save(
                {
                    'epoch': epoch_idx+1,
                    'loss': train_history['loss'],
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                project_dir+"/min_loss.pth"
            )
        if val_flg and (past_history is not None) and ('val_loss' in past_history) and (len(past_history['val_loss']) > 0) and (validation_history['val_loss'] < min(past_history['val_loss'])):
            torch.save(
                {
                    'epoch': epoch_idx+1,
                    'loss': train_history['loss'],
                    'val_loss': validation_history['val_loss'],
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                project_dir+"/min_val_loss.pth"
            )

        # 時間計測
        epoch_time = time.time() - epoch_start_time
        history['epoch_elapsed_time'].append(epoch_time)

        # early stopping
        if es_flg:
            es(train_history['loss'],model)
            if es.early_stop:
                print(f"Stop learning because the loss has risen more than {es_patience} times.")
                break

        # callbackの実行
        if (callbacks is not None) and ('on_epoch_end' in callbacks) and callable(callbacks['on_epoch_end']):
            callbacks['on_epoch_end'](epoch=epoch_idx+1, history_epoch=history_epoch, model=model)

    # callbackの実行
    if (callbacks is not None) and ('on_function_end' in callbacks) and callable(callbacks['on_function_end']):
        callbacks['on_function_end'](model=model, history=history, last_history=history_epoch)

    return history


def train_one_epoch(model, device, train_dataloader, loss_fn, optimizer, callbacks=None, metrics_dict={}):
    """
    pytorchのモデルを1エポックだけ学習する。
    
    Parameters
    ----------
    model : torch.nn.Module
        学習対象のモデル
    device : torch.cuda.device or str
        使用するデバイス
    train_dataloder : torch.utils.data.DataLoader
        学習データ
    loss_fn : torch.nn.lossFunctions
        損失関数
    optimizer : torch.optim.*
        最適化関数
    callback : dict or None
        コールバック関数。実行したい位置によって辞書のkeyを指定する。
            on_train_begin      -> 学習開始前
            on_train_end        -> 学習終了後
    metrics_dict : dict
        計算したいMetricsの辞書を指定。getMetrics関数によって作成されたものをそのまま指定。

    Returns
    -------
    history_epoch : dict
        学習結果
    """

    # 変数
    bar_length = 30 # プログレスバーの長さ
    train_batch_num = len(train_dataloader) # batchの総数

    # metricsからROCは削除
    if 'roc' in metrics_dict:
        metrics_dict.pop('roc')

    # metrics
    metrics = {}
    for k, _ in metrics_dict.items():
        metrics[k] = []

    # callbackの実行
    if (callbacks is not None) and ('on_train_begin' in callbacks) and callable(callbacks['on_train_begin']):
        callbacks['on_train_begin'](model=model)

    # モデルを訓練モードにする
    model.train()

    history_epoch = {}
    losses = []
    t_train = 0

    batch_start_time = time.time()
    for batch_idx, (inputs, labels) in enumerate(train_dataloader):

        metric_batch = {}
        inputs, labels = inputs.to(device), labels.to(device)

        # optimizerを初期化
        optimizer.zero_grad()

        # ニューラルネットワークの処理を行う
        outputs = model(inputs)

        # 損失(出力とラベルとの誤差)の計算
        loss = loss_fn(outputs, labels.float())
        losses.append(loss.item())
        metric_batch['loss'] = loss.item()

        # その他のMetricsの計算
        for k, fn in metrics_dict.items():
            metric = fn(outputs,labels)
            metrics[k].append(metric.item())
            metric_batch[k] = metric.item()

        # 勾配の計算
        loss.backward()
        # for name, param in model.named_parameters():
        #     print(name, param.grad)

        # 重みの更新
        optimizer.step()

        # プログレスバー表示
        interval = time.time() - batch_start_time
        t_train += interval
        eta = str(datetime.timedelta(seconds= int((train_batch_num-batch_idx+1)*interval) ))
        done = math.floor(bar_length * batch_idx / train_batch_num)
        bar = '='*done + ("=" if done == bar_length else ">")  + "."*(bar_length-done)
        print_log(f"\r  \033[K[{bar}] - ETA: {eta:>8}, {math.floor(batch_idx / train_batch_num*100):>3}% ({batch_idx}/{train_batch_num})", line_break=False)

        # 学習状況の表示
        for k,v in metric_batch.items():
            print_log(f", {k.capitalize()}: {v:.04f}", line_break=False)

        batch_start_time = time.time()

    done = bar_length
    bar = '='*done + ("=" if done == bar_length else ">")  + "."*(bar_length-done)
    print_log(f"\r  \033[K[{bar}] - {int(t_train)}s, 100% ({train_batch_num}/{train_batch_num})", line_break=False)

    # 学習状況の表示&保存
    history_epoch['training_elapsed_time'] = t_train
    train_loss = sum(losses) / len(losses)
    print_log(f", Loss: {train_loss:.04f}", line_break=False)
    history_epoch['loss'] = train_loss
    for k,v in metrics.items():
        train_metric = sum(v) / len(v)
        exec('{} = {}'.format(k, train_metric))
        print_log(f", {k.capitalize()}: {train_metric:.04f}", line_break=False)
        history_epoch[k] = train_metric
    print_log()

    # callbackの実行
    if (callbacks is not None) and ('on_train_end' in callbacks) and callable(callbacks['on_train_end']):
        callbacks['on_train_end'](model=model)

    return history_epoch


def validation_one_epoch(model, device, validation_dataloader, loss_fn, callbacks=None, metrics_dict={}):
    """
    pytorchのモデルを1エポックだけ検証する。
    
    Parameters
    ----------
    model : torch.nn.Module
        学習対象のモデル
    device : torch.cuda.device or str
        使用するデバイス
    validation_dataloder : torch.utils.data.DataLoader
        検証データ
    loss_fn : torch.nn.lossFunctions
        損失関数
    callback : dict or None
        コールバック関数。実行したい位置によって辞書のkeyを指定する。
            on_validation_begin -> 検証開始前
            on_validation_end   -> 検証終了後
    metrics_dict : dict
        計算したいMetricsの辞書を指定。getMetrics関数によって作成されたものをそのまま指定。

    Returns
    -------
    history_epoch : dict
        検証結果
    """

    # 変数
    bar_length = 30 # プログレスバーの長さ
    val_batch_num = len(validation_dataloader) # batchの総数

    # metricsからROCは削除
    if 'roc' in metrics_dict:
        metrics_dict.pop('roc')

    # metrics
    val_metrics = {}
    for k, _ in metrics_dict.items():
        val_metrics[k] = []

    # callbackの実行
    if (callbacks is not None) and ('on_validation_begin' in callbacks) and callable(callbacks['on_validation_begin']):
        callbacks['on_validation_begin'](model=model)

    # モデルを評価モードにする
    model.eval()

    history_epoch = {}
    val_losses = []
    t_validation = 0

    with torch.no_grad():

        batch_start_time = time.time()
        for batch_idx, (inputs, labels) in enumerate(validation_dataloader):

            inputs, labels = inputs.to(device), labels.to(device)

            # ニューラルネットワークの処理を行う
            outputs = model(inputs)

            # 損失(出力とラベルとの誤差)の計算
            val_loss = loss_fn(outputs, labels.float())
            val_losses.append(val_loss.item())

            # その他のMetricsの計算
            for k, fn in metrics_dict.items():
                val_metric = fn(outputs,labels)
                val_metrics[k].append(val_metric.item())

            # プログレスバー表示
            interval = time.time() - batch_start_time
            t_validation += interval
            eta = str(datetime.timedelta(seconds= int((val_batch_num-batch_idx+1)*interval) ))
            done = math.floor(bar_length * batch_idx / val_batch_num)
            bar = '='*done + ("=" if done == bar_length else ">")  + "."*(bar_length-done)
            print_log(f"\r  \033[K[{bar}] - ETA: {eta:>8}, {math.floor(batch_idx / val_batch_num*100):>3}% ({batch_idx}/{val_batch_num})", line_break=False)

            batch_start_time = time.time()

        done = bar_length
        bar = '='*done + ("=" if done == bar_length else ">")  + "."*(bar_length-done)
        print_log(f"\r  \033[K[{bar}] - {int(t_validation)}s, 100% ({val_batch_num}/{val_batch_num})", line_break=False)

        # 検証の表示&保存
        history_epoch['val_elapsed_time'] = t_validation
        val_loss = sum(val_losses) / len(val_losses)
        print_log(f", ValLoss: {val_loss:.04f}", line_break=False)
        history_epoch['val_loss'] = val_loss
        for k,v in val_metrics.items():
            val_metric = sum(v) / len(v)
            exec('{} = {}'.format("val_"+k, val_metric))
            print_log(f", Val{k.capitalize()}: {val_metric:.04f}", line_break=False)
            history_epoch["val_"+k] = val_metric
        print_log()

    # callbackの実行
    if (callbacks is not None) and ('on_validation_end' in callbacks) and callable(callbacks['on_validation_end']):
        callbacks['on_validation_end'](model=model)

    return history_epoch


### 評価 ###
def test(model, device, test_dataloader, loss_fn, callbacks=None, metrics_dict={}):
    """
    pytorchのモデルをテストする。
    
    Parameters
    ----------
    model : torch.nn.Module
        テスト対象のモデル
    device : torch.cuda.device or str
        使用するデバイス
    test_dataloder : torch.utils.data.DataLoader
        テストデータ
    loss_fn : torch.nn.lossFunctions
        損失関数
    callback : dict or None
        コールバック関数。実行したい位置によって辞書のkeyを指定する。
            on_test_begin  -> エポック開始前
            on_test_end    -> エポック終了後
    metrics_dict : dict
        計算したいMetricsの辞書を指定。getMetrics関数によって作成されたものをそのまま指定。

    Returns
    -------
    history : dict
        テスト結果
    """

    # 変数
    bar_length = 30 # プログレスバーの長さ
    batch_num = len(test_dataloader) # batchの総数
    history = {} # 返り値用変数

    # モデルを評価モードにする
    model.eval()

    # metrics初期化用
    metrics = {}
    for k, _ in metrics_dict.items():
        if k!='roc':
            metrics[k] = []

    losses = []
    t_test = 0

    # callbackの実行
    if (callbacks is not None) and ('on_test_begin' in callbacks) and callable(callbacks['on_test_begin']):
        callbacks['on_test_begin']()

    print_log(f"Test:")
    with torch.no_grad():

        batch_start_time = time.time()
        for batch_idx, (inputs, labels) in enumerate(test_dataloader):

            inputs, labels = inputs.to(device), labels.to(device)

            # ニューラルネットワークの処理を行う
            outputs = model(inputs)

            # 損失(出力とラベルとの誤差)の計算
            loss = loss_fn(outputs, labels.float())
            losses.append(loss.item())

            # その他のMetricsの計算
            for k, fn in metrics_dict.items():
                if k=='roc':
                    fn.update(outputs,labels.long())
                else:
                    metric = fn(outputs,labels)
                    metrics[k].append(metric.item())

            # プログレスバー表示
            interval = time.time() - batch_start_time
            t_test += interval
            eta = str(datetime.timedelta(seconds= int((batch_num-batch_idx+1)*interval) ))
            done = math.floor(bar_length * batch_idx / batch_num)
            bar = '='*done + ("=" if done == bar_length else ">")  + "."*(bar_length-done)
            print_log(f"\r  \033[K[{bar}] - ETA: {eta:>8}, {math.floor(batch_idx / batch_num*100):>3}% ({batch_idx}/{batch_num})", line_break=False)

            batch_start_time = time.time()

        done = bar_length
        bar = '='*done + ("=" if done == bar_length else ">")  + "."*(bar_length-done)
        print_log(f"\r  \033[K[{bar}] - {int(t_test)}s, 100% ({batch_num}/{batch_num})", line_break=False)

        # 学習状況の表示&保存
        history['test_elapsed_time'] = t_test
        test_loss = sum(losses) / len(losses)
        print_log(f", Loss: {test_loss:.04f}", line_break=False)
        history['loss'] = test_loss
        for k,v in metrics.items():
            if k!='roc':
                test_metric = sum(v) / len(v)
                print_log(f", {k.capitalize()}: {test_metric:.04f}", line_break=False)
                history[k] = test_metric
        if 'roc' in metrics_dict:
            history['roc'] = metrics_dict['roc'].compute()
        print_log()

    # callbackの実行
    if (callbacks is not None) and ('on_test_end' in callbacks) and callable(callbacks['on_test_end']):
        callbacks['on_test_end']()

    return history





### 損失関数取得 ###
def getCriterion(class_weights, device):
    weight = torch.tensor(list(class_weights.values()), dtype=torch.float).to(device)
    if len(weight)==2:
        weight[1] = weight[1] / weight[0]
        weight[0] = weight[0] / weight[0]
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight[1])
        # criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=weight)
        # criterion = nn.NLLLoss()
    return criterion


# ### ここから ###

# ### Metricsリスト取得 ###
# def getMetrics(device, mode='all', num_classes=2, average='micro'):
#     """
#     Metricsの辞書リストを取得する
    
#     Parameters
#     ----------
#     mode : str or list
#         取得したいMetricsの文字列、またはそのリストを指定。
#             'all'
#             'accuracy'
#     num_classes : int
#         ラベル数の指定
#     average : str or None
#         平均化のタイプを指定。以下tensorflowによる説明
#             None        -> 平均化は実行されず、各クラスのスコアが返される。
#             'micro'     -> TP,FP,TN,FNなどの合計を数えることによって、グローバルに計算する。
#             'macro'     -> 各ラベルのメトリックを計算し、重み付けされていない平均を返す。これはラベルの不均衡を考慮していない。
#             'weighted'  -> 各ラベルのメトリックを計算し、各ラベルの真のインスタンス数によって重み付けされた平均を返す。

#     Returns
#     -------
#     metrics_dict : dict
#         Metricsの辞書リスト
#     """

#     task = "binary" if num_classes==2 else "multilabel"
#     metrics_dict = {}
#     if mode=='all' or ('all' in mode) or mode=='accuracy' or ('accuracy' in mode):
#         metrics_dict['accuracy'] = torchmetrics.Accuracy(task=task,num_labels=num_classes).to(device)
#     if mode=='all' or ('all' in mode) or mode=='auc' or ('auc' in mode):
#         metrics_dict['auc'] = torchmetrics.AUROC(task=task,num_labels=num_classes).to(device)
#     if mode=='all' or ('all' in mode) or mode=='precision' or ('precision' in mode):
#         metrics_dict['precision'] = torchmetrics.Precision(task=task,num_labels=num_classes).to(device)
#     if mode=='all' or ('all' in mode) or mode=='recall' or ('recall' in mode):
#         metrics_dict['recall'] = torchmetrics.Recall(task=task,num_labels=num_classes).to(device)
#     if mode=='all' or ('all' in mode) or mode=='specificity' or ('specificity' in mode):
#         metrics_dict['specificity'] = torchmetrics.Specificity(task=task,num_labels=num_classes).to(device)
#     if mode=='all' or ('all' in mode) or mode=='f1' or ('f1' in mode):
#         metrics_dict['f1'] = torchmetrics.F1Score(task=task,num_labels=num_classes).to(device)
#     if mode=='all' or ('all' in mode) or mode=='roc' or ('roc' in mode):
#         metrics_dict['roc'] = torchmetrics.ROC(task=task,num_labels=num_classes).to(device)
#     return metrics_dict


# ### transform作成 ###
# def getTransforms(
#         image_size=None,#(3,256,256),
#         normalization=False,
#         rotation_range=0.0,
#         width_shift_range=0.0,
#         height_shift_range=0.0,
#         brightness_range=0.0,
#         shear_range=0.0,
#         zoom_range=0.0,
#         horizontal_flip=False,
#         vertical_flip=False
#     ):
#     """
#     画像の前処理インスタンスを取得する
#     """

#     transform_list = [
#         torchvision.transforms.ToTensor(),  # テンソル化 & 正規化
#     ]

#     # グレースケール
#     if (image_size is not None) and (image_size[0]==1):
#         transform_list.append(torchvision.transforms.Grayscale())

#     # 画像サイズ
#     if image_size is not None:
#         transform_list.append(torchvision.transforms.Resize(image_size[1:]))

#     # 0〜1 → -1〜1
#     if normalization:
#         transform_list.append(torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

#     # 回転
#     if rotation_range != 0:
#         transform_list.append(torchvision.transforms.RandomRotation(rotation_range, expand=False))

#     # シフト
#     if width_shift_range != 0 or height_shift_range != 0:
#         transform_list.append(torchvision.transforms.RandomAffine(0, translate=(width_shift_range,height_shift_range)))

#     # 明るさ
#     if brightness_range != 0:
#         transform_list.append(torchvision.transforms.ColorJitter())

#     # せん断 (四角形→平行四辺形)
#     if shear_range != 0:
#         transform_list.append(torchvision.transforms.RandomAffine(0, shear=shear_range))

#     # 拡大
#     if zoom_range != 0:
#         transform_list.append(torchvision.transforms.RandomAffine(0, scale=(1-zoom_range,1+zoom_range)))

#     # 左右反転
#     if horizontal_flip != 0:
#         transform_list.append(torchvision.transforms.RandomHorizontalFlip())

#     # 上下反転
#     if vertical_flip != 0:
#         transform_list.append(torchvision.transforms.RandomVerticalFlip())

#     return torchvision.transforms.Compose(transform_list)

# ### ここまで ###



### 結果保存 ###
def saveHistory(history_dict, history_path="./history.json"):

    # historyがなければ新規作成
    if not os.path.isfile(history_path):
        history_empty = {}
        for k,v in history_dict.items():
            history_empty[k] = []
        with open(history_path, 'w') as f:
            json.dump(history_empty, f, indent=2, sort_keys=False, ensure_ascii=False)

    # history読み込み
    with open(history_path, 'r') as f:
        history = json.load(f)

    # history更新
    new_list = [None] * len(history[list(history.keys())[0]])
    for k,v in history_dict.items():
        if k in history:
            history[k].append(v)
        else:
            history[k] = copy.deepcopy(new_list).append(v)

    # history保存
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, sort_keys=False, ensure_ascii=False)


### historyの差異合わせ ###
def adjustHistory(epoch, history_path="./history.json"):

    # history読み込み
    with open(history_path, 'r') as f:
        history = json.load(f)

    # 差異合わせ
    for k,_ in history.items():
        if len(history[k]) > epoch:
            history[k] = history[k][:epoch]

    # history保存
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, sort_keys=False, ensure_ascii=False)


### 結果描画 ###
def saveLossGraph(graph_data, save_path='result.png', title='Model accuracy'):
    plt.clf()
    mpl_color_list = ["blue","orange","green","red","purple","brown","pink","gray","olive","cyan"]
    for i, (metrics_name, data) in enumerate(graph_data.items()):
        plt.plot(range(1, len(data)+1), data, color=mpl_color_list[i], linestyle='solid', marker='o', label=metrics_name)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.clf()


### ROC曲線描画 ###
def saveRocCurve(roc_data, save_path='roc.png', title='ROC Curve'):
    plt.clf()
    mpl_color_list = ["blue","orange","green","red","purple","brown","pink","gray","olive","cyan"]
    fpr, tpr, _thresholds = roc_data
    if type(fpr) == list:
        for i in range(len(fpr)):
            fpr[i] = fpr[i].cpu().numpy()
            tpr[i] = tpr[i].cpu().numpy()
            plt.plot(fpr[i], tpr[i], color=mpl_color_list[i], linestyle='solid', marker='', label=f'label {i}')
    else:
        fpr = fpr.cpu().numpy()
        tpr = tpr.cpu().numpy()
        plt.plot(fpr, tpr, color=mpl_color_list[0], linestyle='solid', marker='')
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(save_path)
    plt.clf()


### モデルサイズ取得 ###
def get_model_memory_size(model):
    type_sizes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.float64: 8
    }
    total_memory_size = 0
    for p in model.parameters():
        total_memory_size += p.numel() * type_sizes[p.dtype]
    return total_memory_size


### jsonパラメータ保存 ###
def saveParams(params,filename="./params.json"):
    with open(filename, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=False, ensure_ascii=False)


### jsonパラメータ読込 ###
def loadParams(filename="./params.json"):
    with open(filename, 'r') as f:
        params = json.load(f)
    return params


### jsonパラメータ trained_epochsインクリメント ###
def incrementTrainedEpochs(params_path="./params.json"):
    with open(params_path, 'r') as f:
        params = json.load(f)
    params['trained_epochs'] += 1
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=False, ensure_ascii=False)


### EarlyStopping ###
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


### ログ表示 ###
def print_log(message="", line_break=True):
    if line_break:
        sys.stdout.write(message + "\n")
    else:
        sys.stdout.write(message)
    sys.stdout.flush()


### 画像表示 ###
def dispImages(dataloader=None, file_path='./image_sample.jpg', model=None, classes=['Celeb-real-image-face-90', 'Celeb-synthesis-image-face-90']):
    plt.clf()
    col = 5
    row = math.ceil(len(dataloader)/col)
    fig, ax = plt.subplots(nrows=row, ncols=col, figsize=(col*2,row*2+1))
    fig.suptitle("Image Sample", fontsize=20, color='black')
    for image_idx, (image, label) in enumerate(dataloader):
        if model is not None:
            model.eval()
            with torch.no_grad():
                output = model(image)
                if len(classes) == 2:
                    predict = output[0].to('cpu').detach().numpy().copy()[0]
                    predict_label = 1 if output[0].to('cpu').detach().numpy().copy()[0] >= 0.5 else 0
                else:
                    predict = np.max(output[0].to('cpu').detach().numpy().copy())
                    predict_label = np.argmax(output[0].to('cpu').detach().numpy().copy())
        image = image[0].permute(1,2,0).to('cpu').detach().numpy().copy()
        _r= image_idx//col
        _c= image_idx%col
        # mpl_color_list_str = ["blue","red","green","brown","orange","purple","pink","gray","olive","cyan"]
        # str_color = mpl_color_list_str[label.item()]
        str_color = "red" if label.item()==predict_label else "blue"
        ax[_r,_c].set_title(f"Label:{label.item()}, Pred: {predict_label}({predict:.4f})", fontsize=8, color=str_color)
        ax[_r,_c].axes.xaxis.set_visible(False)
        ax[_r,_c].axes.yaxis.set_visible(False)
        ax[_r,_c].imshow(image)
    plt.savefig(file_path,dpi=100)
    plt.clf()


### 何枚かの画像をテストして表示 ###
def saveTest(
        data_num=10,
        file_path='./image_sample.jpg',
        model=None,
        data_dir=os.getenv('FAKE_DATA_PATH'),
        classes=['Celeb-real-image-face-90', 'Celeb-synthesis-image-face-90'],
        transform=getTransforms(),
        validation_rate=0.1,
        test_rate=0.1
    ):
    disp_dataloader = getMiniCelebDataLoader(
        data_num=data_num,
        data_dir=data_dir,
        classes=classes,
        transform=transform,
        validation_rate=validation_rate,
        test_rate=test_rate
    )
    dispImages(
        dataloader=disp_dataloader,
        file_path=file_path,
        model=model,
        classes=classes
    )
    return


### ヒートマップ表示(動作しない) ###
def saveHeatMap(
        model,
        # device,
        data_num=4,
        file_path='./image_sample.jpg',
        data_dir=os.getenv('FAKE_DATA_PATH'),
        classes=['Celeb-real-image-face-90', 'Celeb-synthesis-image-face-90'],
        transform=getTransforms(),
        validation_rate=0.1,
        test_rate=0.1
    ):

    plt.clf()

    disp_dataloader = getMiniCelebDataLoader(
        data_num=data_num,
        data_dir=data_dir,
        classes=classes,
        transform=transform,
        validation_rate=validation_rate,
        test_rate=test_rate
    )

    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam import GradCAM

    model.eval()
    with torch.no_grad():
        # target_layer = [model.modules().conv1]
        target_layer = get_layer_list(model)[2]
        cam = GradCAM(
            model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available()
        )

        col = 2
        row = math.ceil(len(disp_dataloader)/col)
        fig, ax = plt.subplots(nrows=row, ncols=col*2, figsize=(col*2*2,row*2+1))
        fig.suptitle("HeatMap", fontsize=20, color='black')

        for image_idx, (image, label) in enumerate(disp_dataloader):
            grayscale_cam = cam(
                input_tensor=image,
                # targets=[ClassifierOutputTarget(label)] if len(classes)>2 else [BinaryClassifierOutputTarget(label)],
                targets=[ClassifierOutputTarget(torch.argmax(label, dim=1).item())] if len(classes)>2 else [BinaryClassifierOutputTarget(label.item())],
            )

            _r= image_idx//col
            _c_i= (image_idx%col)*2
            _c_h= (image_idx%col)*2+1
            # 通常画像プロット
            ax[_r,_c_i].set_title(f"label: {label.item()}", fontsize=16, color="black")
            ax[_r,_c_i].axes.xaxis.set_visible(False)
            ax[_r,_c_i].axes.yaxis.set_visible(False)
            ax[_r,_c_i].imshow(image.permute(1, 2, 0).numpy())
            # ヒートマッププロット
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(image.permute(1, 2, 0).numpy(), grayscale_cam, use_rgb=True)
            ax[_r,_c_h].axes.xaxis.set_visible(False)
            ax[_r,_c_h].axes.yaxis.set_visible(False)
            ax[_r,_c_h].imshow(visualization)
    plt.clf()
    return


### batch_sizeの見積もり(なんかめちゃ時間かかるから未使用) ###
def estimate_batch_size(model, device, dataloader, gpu_memory_limit=31.0, num_gpus=4):

    # ダミーデータを生成してバッチサイズを調整する
    batch_size = 1
    dummy_input = torch.randn(batch_size, *dataloader.dataset[0][0].shape).to(device)
    model = model.to(device)

    # バッチサイズを増やしながらGPUのメモリ容量を確認する
    while True:
        sys.stdout.write(str(batch_size) + "\n")
        try:
            # バッチサイズを増やしてモデルを実行する
            dummy_loader = DataLoader(dataloader.dataset, batch_size=batch_size, shuffle=False)
            with torch.no_grad():
                for inputs, _ in dummy_loader:
                    inputs = inputs.to(device)
                    _ = model(inputs)

            # GPUのメモリ使用量を取得する
            memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB

            # バッチサイズが制約に達したら終了する
            if memory_allocated > gpu_memory_limit * num_gpus:
                break

            # バッチサイズを増やす
            batch_size *= 2

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                break
            else:
                raise

    # バッチサイズをGPU枚数で割る
    batch_size //= num_gpus

    return batch_size

### batch_sizeの取得(RAIDEN)(32GB/GPU) ###
def get_batch_size_per_gpu_raiden_celeb(structure_name):
    bs_dict = {
        "XceptionNet": 128,
        "Vgg16": 128,
        "DenseNet201": 64, #未検証
        "EfficientNetV2L": 32,
        "EfficientNetV2M": 64, #未検証
        "EfficientNetV2S": 128,
        "MnasNet": 128,
    }
    if structure_name in bs_dict:
        return bs_dict[structure_name]
    return 32

### batch_sizeの取得(GHPC)(12GB/GPU) ###
def get_batch_size_per_gpu_ghpc(structure_name):
    bs_dict = {
        "Vgg16": 32,
    }
    if structure_name in bs_dict:
        return bs_dict[structure_name]
    return 32    

####################################################################################################






















# (時間計測)
def measureTime(fn):
    t = time.time()
    fn()
    return time.time() - t





























####################################################################################################
# モデル構造
####################################################################################################

# SampleCnn (MNIST用)
@model_wrapper
class SampleCnn(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        last_dim = 1 if num_classes==2 else num_classes

        self.c1 = nn.Conv2d(in_channels, 6, kernel_size=5, padding=2)
        self.r1 = nn.ReLU(inplace=True)
        self.m1 = nn.MaxPool2d(2)
        self.c2 = nn.Conv2d(6, 16, kernel_size=5)
        self.r2 = nn.ReLU(inplace=True)
        self.m2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*5*5, 120)
        self.r3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(120,84)
        self.r4 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(84, last_dim)
        self.last_activation = nn.Softmax(dim=1) if num_classes>2 else nn.Sigmoid()

        # # weight init                                                                      
        # for m in self.children():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.m1(self.r1(self.c1(x)))
        x = self.m2(self.r2(self.c2(x)))
        x = self.flatten(x)
        x = self.r3(self.fc1(x))
        x = self.r4(self.fc2(x))
        x = self.last_activation(self.fc3(x))
        return x




### modelのlayer_list取得 ###
def get_layer_list(model, withName=False):
    layer_list = []
    for idx, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            if withName:
                layer_list.append((name, module))
            else:
                layer_list.append(module)
        elif (idx != 0) and isinstance(module, torch.nn.Module):
            sub_layer_list = get_layer_list(module)
            if withName:
                layer_list.extend([(f"{name}.{sub_name}", sub_module) for sub_name, sub_module in sub_layer_list])
            else:
                layer_list.extend([sub_module for sub_module in sub_layer_list])
    return layer_list

### modelのパラメータ初期化 ###
def initialize_parameters(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            # init.xavier_uniform_(m.weight) # Xavierの初期値
            init.kaiming_uniform_(m.weight) # Heの初期値 (活性化関数がReLUの場合)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)


### VGGNet ###
def Vgg16(in_channels=1, num_classes=10, pretrained=False):
    model = torchvision.models.vgg16(pretrained=pretrained, progress=False)
    last_dim = 1 if num_classes==2 else num_classes
    # 開始層の変更
    old_conv = model.features[0]
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False if (old_conv.bias is None) else True
    )
    model.features[0] = new_conv
    # 最終層の変更
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, last_dim)
    # パラメータの初期化
    initialize_parameters(model)
    return model


### InceptionNet ###
def InceptionV3(in_channels=1, num_classes=10, pretrained=False):
    model = torchvision.models.inception_v3(pretrained=pretrained, progress=False)
    last_dim = 1 if num_classes==2 else num_classes
    # 開始層の変更
    old_conv = model.Conv2d_1a_3x3.conv
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False if (old_conv.bias is None) else True
    )
    model.Conv2d_1a_3x3.conv = new_conv
    # 最終層の変更
    model.fc = nn.Linear(model.fc.in_features, last_dim)
    # パラメータの初期化
    initialize_parameters(model)
    return model


### ResNet ###
def ResNet18(in_channels=1, num_classes=10, pretrained=False):
    model = torchvision.models.resnet18(pretrained=pretrained, progress=False)
    last_dim = 1 if num_classes==2 else num_classes
    # 開始層の変更
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False if (old_conv.bias is None) else True
    )
    model.conv1 = new_conv
    # 最終層の変更
    model.fc = nn.Linear(model.fc.in_features, last_dim)
    # パラメータの初期化
    initialize_parameters(model)
    return model

def ResNet50(in_channels=1, num_classes=10, pretrained=False):
    model = torchvision.models.resnet50(pretrained=pretrained, progress=False)
    last_dim = 1 if num_classes==2 else num_classes
    # 開始層の変更
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False if (old_conv.bias is None) else True
    )
    model.conv1 = new_conv
    # 最終層の変更
    model.fc = nn.Linear(model.fc.in_features, last_dim)
    # パラメータの初期化
    initialize_parameters(model)
    return model

def ResNet101(in_channels=1, num_classes=10, pretrained=False):
    model = torchvision.models.resnet101(pretrained=pretrained, progress=False)
    last_dim = 1 if num_classes==2 else num_classes
    # 開始層の変更
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False if (old_conv.bias is None) else True
    )
    model.conv1 = new_conv
    # 最終層の変更
    model.fc = nn.Linear(model.fc.in_features, last_dim)
    # パラメータの初期化
    initialize_parameters(model)
    return model

def ResNet152(in_channels=1, num_classes=10, pretrained=False):
    model = torchvision.models.resnet152(pretrained=pretrained, progress=False)
    last_dim = 1 if num_classes==2 else num_classes
    # 開始層の変更
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False if (old_conv.bias is None) else True
    )
    model.conv1 = new_conv
    # 最終層の変更
    model.fc = nn.Linear(model.fc.in_features, last_dim)
    # パラメータの初期化
    initialize_parameters(model)
    return model

def WideResNet50_2(in_channels=1, num_classes=10, pretrained=False):
    model = torchvision.models.wide_resnet50_2(pretrained=pretrained, progress=False)
    last_dim = 1 if num_classes==2 else num_classes
    # 開始層の変更
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False if (old_conv.bias is None) else True
    )
    model.conv1 = new_conv
    # 最終層の変更
    model.fc = nn.Linear(model.fc.in_features, last_dim)
    # パラメータの初期化
    initialize_parameters(model)
    return model


### DenseNet ###
def DenseNet121(in_channels=1, num_classes=10, pretrained=False):
    model = torchvision.models.densenet121(pretrained=pretrained, progress=False)
    last_dim = 1 if num_classes==2 else num_classes
    # 開始層の変更
    old_conv = model.features.conv0
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False if (old_conv.bias is None) else True
    )
    model.features.conv0 = new_conv
    # 最終層の変更
    model.classifier = nn.Linear(model.classifier.in_features, last_dim)
    # パラメータの初期化
    initialize_parameters(model)
    return model

def DenseNet161(in_channels=1, num_classes=10, pretrained=False):
    model = torchvision.models.densenet161(pretrained=pretrained, progress=False)
    last_dim = 1 if num_classes==2 else num_classes
    # 開始層の変更
    old_conv = model.features.conv0
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False if (old_conv.bias is None) else True
    )
    model.features.conv0 = new_conv
    # 最終層の変更
    model.classifier = nn.Linear(model.classifier.in_features, last_dim)
    # パラメータの初期化
    initialize_parameters(model)
    return model

def DenseNet169(in_channels=1, num_classes=10, pretrained=False):
    model = torchvision.models.densenet169(pretrained=pretrained, progress=False)
    last_dim = 1 if num_classes==2 else num_classes
    # 開始層の変更
    old_conv = model.features.conv0
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False if (old_conv.bias is None) else True
    )
    model.features.conv0 = new_conv
    # 最終層の変更
    model.classifier = nn.Linear(model.classifier.in_features, last_dim)
    # パラメータの初期化
    initialize_parameters(model)
    return model

def DenseNet201(in_channels=1, num_classes=10, pretrained=False):
    model = torchvision.models.densenet201(pretrained=pretrained, progress=False)
    last_dim = 1 if num_classes==2 else num_classes
    # 開始層の変更
    old_conv = model.features.conv0
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False if (old_conv.bias is None) else True
    )
    model.features.conv0 = new_conv
    # 最終層の変更
    model.classifier = nn.Linear(model.classifier.in_features, last_dim)
    # パラメータの初期化
    initialize_parameters(model)
    return model


### GoogleNet ###
def GoogleNet(in_channels=1, num_classes=10, pretrained=False):
    model = torchvision.models.googlenet(pretrained=pretrained, progress=False)
    last_dim = 1 if num_classes==2 else num_classes
    # 開始層の変更
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False if (old_conv.bias is None) else True
    )
    model.conv1 = new_conv
    # 最終層の変更
    model.fc = nn.Linear(model.fc.in_features, last_dim)
    # パラメータの初期化
    initialize_parameters(model)
    return model


### MobileNet ###
def MobileNetV2(in_channels=1, num_classes=10, pretrained=False):
    model = torchvision.models.mobilenet_v2(pretrained=pretrained, progress=False)
    last_dim = 1 if num_classes==2 else num_classes
    # 開始層の変更
    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False if (old_conv.bias is None) else True
    )
    model.features[0][0] = new_conv
    # 最終層の変更
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, last_dim)
    # パラメータの初期化
    initialize_parameters(model)
    return model

def MobileNetV3S(in_channels=1, num_classes=10, pretrained=False):
    model = torchvision.models.mobilenet_v3_small(pretrained=pretrained, progress=False)
    last_dim = 1 if num_classes==2 else num_classes
    # 開始層の変更
    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False if (old_conv.bias is None) else True
    )
    model.features[0][0] = new_conv
    # 最終層の変更
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, last_dim)
    # パラメータの初期化
    initialize_parameters(model)
    return model

def MobileNetV3L(in_channels=1, num_classes=10, pretrained=False):
    model = torchvision.models.mobilenet_v3_large(pretrained=pretrained, progress=False)
    last_dim = 1 if num_classes==2 else num_classes
    # 開始層の変更
    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False if (old_conv.bias is None) else True
    )
    model.features[0][0] = new_conv
    # 最終層の変更
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, last_dim)
    # パラメータの初期化
    initialize_parameters(model)
    return model


### MNASNet ###
def MnasNet(in_channels=1, num_classes=10, pretrained=False):
    model = torchvision.models.mnasnet1_0(pretrained=pretrained, progress=False)
    last_dim = 1 if num_classes==2 else num_classes
    # 開始層の変更
    old_conv = model.layers[0]
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False if (old_conv.bias is None) else True
    )
    model.layers[0] = new_conv
    # 最終層の変更
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, last_dim)
    # パラメータの初期化
    initialize_parameters(model)
    return model


### EfficientNet ###
def EfficientNetB7(in_channels=1, num_classes=10, pretrained=False):
    model = torchvision.models.efficientnet_b7(pretrained=pretrained, progress=False)
    last_dim = 1 if num_classes==2 else num_classes
    # 開始層の変更
    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False if (old_conv.bias is None) else True
    )
    model.features[0][0] = new_conv
    # 最終層の変更
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, last_dim)
    # パラメータの初期化
    initialize_parameters(model)
    return model

def EfficientNetV2S(in_channels=1, num_classes=10, pretrained=False):
    model = torchvision.models.efficientnet_v2_s(pretrained=pretrained, progress=False)
    last_dim = 1 if num_classes==2 else num_classes
    # 開始層の変更
    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False if (old_conv.bias is None) else True
    )
    model.features[0][0] = new_conv
    # 最終層の変更
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, last_dim)
    # パラメータの初期化
    initialize_parameters(model)
    return model

def EfficientNetV2M(in_channels=1, num_classes=10, pretrained=False):
    model = torchvision.models.efficientnet_v2_m(pretrained=pretrained, progress=False)
    last_dim = 1 if num_classes==2 else num_classes
    # 開始層の変更
    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False if (old_conv.bias is None) else True
    )
    model.features[0][0] = new_conv
    # 最終層の変更
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, last_dim)
    # パラメータの初期化
    initialize_parameters(model)
    return model

def EfficientNetV2L(in_channels=1, num_classes=10, pretrained=False):
    model = torchvision.models.efficientnet_v2_l(pretrained=pretrained, progress=False)
    last_dim = 1 if num_classes==2 else num_classes
    # 開始層の変更
    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False if (old_conv.bias is None) else True
    )
    model.features[0][0] = new_conv
    # 最終層の変更
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, last_dim)
    # パラメータの初期化
    initialize_parameters(model)
    return model


### ViT (入力を224にしないと使用できない) ###
def ViT_B16(in_channels=1, num_classes=10, pretrained=False):
    model = torchvision.models.vit_b_16(pretrained=pretrained, progress=False)
    last_dim = 1 if num_classes==2 else num_classes
    # 開始層の変更
    old_conv = model.conv_proj
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False if (old_conv.bias is None) else True
    )
    model.conv_proj = new_conv
    # 最終層の変更
    model.heads.head = nn.Linear(model.heads.head.in_features, last_dim)
    # パラメータの初期化
    initialize_parameters(model)
    return model

def ViT_B32(in_channels=1, num_classes=10, pretrained=False):
    model = torchvision.models.vit_b_32(pretrained=pretrained, progress=False)
    last_dim = 1 if num_classes==2 else num_classes
    # 開始層の変更
    old_conv = model.conv_proj
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False if (old_conv.bias is None) else True
    )
    model.conv_proj = new_conv
    # 最終層の変更
    model.heads.head = nn.Linear(model.heads.head.in_features, last_dim)
    # パラメータの初期化
    initialize_parameters(model)
    return model

def ViT_L16(in_channels=1, num_classes=10, pretrained=False):
    model = torchvision.models.vit_l_16(pretrained=pretrained, progress=False)
    last_dim = 1 if num_classes==2 else num_classes
    # 開始層の変更
    old_conv = model.conv_proj
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False if (old_conv.bias is None) else True
    )
    model.conv_proj = new_conv
    # 最終層の変更
    model.heads.head = nn.Linear(model.heads.head.in_features, last_dim)
    # パラメータの初期化
    initialize_parameters(model)
    return model

def ViT_L32(in_channels=1, num_classes=10, pretrained=False):
    model = torchvision.models.vit_l_32(pretrained=pretrained, progress=False)
    last_dim = 1 if num_classes==2 else num_classes
    # 開始層の変更
    old_conv = model.conv_proj
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False if (old_conv.bias is None) else True
    )
    model.conv_proj = new_conv
    # 最終層の変更
    model.heads.head = nn.Linear(model.heads.head.in_features, last_dim)
    # パラメータの初期化
    initialize_parameters(model)
    return model

def ViT_H14(in_channels=1, num_classes=10, pretrained=False):
    model = torchvision.models.vit_h_14(pretrained=pretrained, progress=False)
    last_dim = 1 if num_classes==2 else num_classes
    # 開始層の変更
    old_conv = model.conv_proj
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False if (old_conv.bias is None) else True
    )
    model.conv_proj = new_conv
    # 最終層の変更
    model.heads.head = nn.Linear(model.heads.head.in_features, last_dim)
    # パラメータの初期化
    initialize_parameters(model)
    return model


### LightInceptionNetV1 ###
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False, padding_mode='replicate'):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias, padding_mode=padding_mode
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class LightInceptionModuleV1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LightInceptionModuleV1, self).__init__()
        cell_num = 8

        self.cell1_pool = nn.AvgPool2d(kernel_size=(2,2), stride=1, padding=(2//2, 2//2))
        self.cell1_conv = SeparableConv2d(in_channels, out_channels // cell_num, (3,3), stride=1, padding=(3//2, 3//2))
        self.easy = SeparableConv2d(in_channels, out_channels // cell_num, (1,1), stride=1, padding=(1//2, 1//2))
        self.cell3 = SeparableConv2d(out_channels // cell_num, out_channels // cell_num, (3,3), stride=1, padding=(3//2, 3//2))
        self.cell4 = SeparableConv2d(out_channels // cell_num, out_channels // cell_num, (3,3), stride=1, padding=(3//2, 3//2))
        self.cell5 = nn.Sequential(
            SeparableConv2d(out_channels // cell_num, out_channels // cell_num, (3,3), stride=1, padding=(3//2, 3//2)),
            SeparableConv2d(out_channels // cell_num, out_channels // cell_num, (3,3), stride=1, padding=(3//2, 3//2))
        )
        self.cell6 = nn.Sequential(
            SeparableConv2d(out_channels // cell_num, out_channels // cell_num, (3,3), stride=1, padding=(3//2, 3//2)),
            SeparableConv2d(out_channels // cell_num, out_channels // cell_num, (3,3), stride=1, padding=(3//2, 3//2))
        )
        self.cell7 = nn.Sequential(
            SeparableConv2d(out_channels // cell_num, out_channels // cell_num, (3,3), stride=1, padding=(3//2, 3//2)),
            SeparableConv2d(out_channels // cell_num, out_channels // cell_num, (3,3), stride=1, padding=(3//2, 3//2)),
            SeparableConv2d(out_channels // cell_num, out_channels // cell_num, (3,3), stride=1, padding=(3//2, 3//2))
        )
        self.cell8_1 = SeparableConv2d(out_channels // cell_num, out_channels // 2 // cell_num, (3,3), stride=1, padding=(3//2, 3//2))
        self.cell8_residual1 = SeparableConv2d(out_channels // cell_num, out_channels // 2 // cell_num, (1,1), stride=1, padding=(1//2, 1//2))
        self.cell8_2 = SeparableConv2d(out_channels // cell_num, out_channels // 2 // cell_num, (3,3), stride=1, padding=(3//2, 3//2))
        self.cell8_residual2 = SeparableConv2d(out_channels // cell_num, out_channels // 2 // cell_num, (1,1), stride=1, padding=(1//2, 1//2))
        self.cell8_3 = SeparableConv2d(out_channels // cell_num, out_channels // 2 // cell_num, (3,3), stride=1, padding=(3//2, 3//2))
        self.cell8_residual3 = SeparableConv2d(out_channels // cell_num, out_channels // 2 // cell_num, (1,1), stride=1, padding=(1//2, 1//2))

    def forward(self, x):
        cell1 = self.cell1_pool(x)
        cell1 = self.cell1_conv(x)
        easy = self.easy(x)
        cell2 = easy
        cell3 = self.cell3(easy)
        cell4 = self.cell4(easy)
        cell5 = self.cell5(easy)
        cell6 = self.cell6(easy)
        cell7 = self.cell7(easy)
        cell8_1 = self.cell8_1(easy)
        cell8_residual1 = self.cell8_residual1(easy)
        cell8_1 = torch.cat([cell8_residual1, cell8_1], dim=1)
        cell8_2 = self.cell8_2(cell8_1)
        cell8_residual2 = self.cell8_residual2(cell8_1)
        cell8_2 = torch.cat([cell8_residual2, cell8_2], dim=1)
        cell8_3 = self.cell8_3(cell8_2)
        cell8_residual3 = self.cell8_residual3(cell8_2)
        cell8_3 = torch.cat([cell8_residual3, cell8_3], dim=1)
        y = torch.cat([cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8_3], dim=1)
        return y

class LightInceptionBlockV1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LightInceptionBlockV1, self).__init__()

        self.red = nn.Sequential(
            SeparableConv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=1//2),
            nn.BatchNorm2d(out_channels)
        )
        self.conv1 = nn.Sequential(
            SeparableConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            SeparableConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=(3//2, 3//2))
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        red = self.red(x)
        x = self.conv1(x)
        x = self.conv2(x)
        y = x + red
        return y

class LightInceptionNetV1(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):

        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        last_dim = 1 if num_classes==2 else num_classes
        repetition_num = 5

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            SeparableConv2d(in_channels, 32, kernel_size=3, stride=2, padding=3//2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            SeparableConv2d(32, 64, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.block1 = LightInceptionBlockV1(64,128)
        self.block2 = LightInceptionBlockV1(128,256)
        self.block3 = LightInceptionBlockV1(256,512)
        self.conv3 = SeparableConv2d(512, 512, kernel_size=3, stride=1, padding=3//2)
        repetition_list = []
        for i in range(repetition_num):
            layer_list = []
            layer_list.append(nn.BatchNorm2d(512))
            layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(LightInceptionModuleV1(512,512))
            layer_list.append(nn.BatchNorm2d(512))
            layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(LightInceptionModuleV1(512,512))
            layer_list.append(nn.BatchNorm2d(512))
            layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.Dropout(0.3))
            layer_list.append(LightInceptionModuleV1(512,512))
            repetition_list.append(layer_list)
        self.repetition_list = repetition_list
        self.bn4 = nn.BatchNorm2d(512)
        self.block4 = LightInceptionBlockV1(512,768)
        self.conv5 = nn.Sequential(
            LightInceptionModuleV1(768,768),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            LightInceptionModuleV1(768,768),
            nn.BatchNorm2d(768)
        )
        self.block5 = LightInceptionBlockV1(768,1024)
        self.conv7 = nn.Sequential(
            SeparableConv2d(1024, 1536, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True)
        )
        self.conv8 = nn.Sequential(
            SeparableConv2d(1536, 2048, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv9 = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.conv3(x)
        for block in self.repetition_list:
            red = x
            for layer in block:
                x = layer(x)
            x = red + x
        x = self.bn4(x)
        x = self.relu(x)
        x = self.block4(x)
        red = x
        x = self.conv5(x)
        x = self.conv6(x)
        x = red + x
        x = self.relu(x)
        x = self.block5(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.global_avg_pool(x)
        x = x.squeeze(dim=3).squeeze(dim=2)
        x = self.conv9(x)
        x = self.fc(x)
        return x















class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


@model_wrapper
class XceptionNet(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, in_channels=1, num_classes=10):

        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        last_dim = 1 if num_classes==2 else num_classes

        self.conv1 = nn.Conv2d(in_channels, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, last_dim)
        self.last_activation = nn.Softmax(dim=1) if num_classes>2 else nn.Sigmoid()
        
        initialize_parameters(self)

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)

        return x

    def logits(self, features):
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = self.last_activation(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


























print(f"FINISH IMPORT: {now()}")
print("\n\n")
