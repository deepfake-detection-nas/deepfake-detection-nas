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




GPU_COUNT = torch.cuda.device_count()
if torch.cuda.is_available():
    print("ALL GPU COUNT: "+str(torch.cuda.device_count()))
    print("ALL GPU NAMES: "+str([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))
print()










### transform作成 ###
def getTransforms(
        image_size=None,#(3,256,256),
        normalization=False,
        rotation_range=0.0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        brightness_range=0.0,
        shear_range=0.0,
        zoom_range=0.0,
        horizontal_flip=False,
        vertical_flip=False
    ):
    """
    画像の前処理インスタンスを取得する
    """

    transform_list = [
        torchvision.transforms.ToTensor(),  # テンソル化 & 正規化
    ]

    # グレースケール
    if (image_size is not None) and (image_size[0]==1):
        transform_list.append(torchvision.transforms.Grayscale())

    # 画像サイズ
    if image_size is not None:
        transform_list.append(torchvision.transforms.Resize(image_size[1:]))

    # 0〜1 → -1〜1
    if normalization:
        transform_list.append(torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    # 回転
    if rotation_range != 0:
        transform_list.append(torchvision.transforms.RandomRotation(rotation_range, expand=False))

    # シフト
    if width_shift_range != 0 or height_shift_range != 0:
        transform_list.append(torchvision.transforms.RandomAffine(0, translate=(width_shift_range,height_shift_range)))

    # 明るさ
    if brightness_range != 0:
        transform_list.append(torchvision.transforms.ColorJitter())

    # せん断 (四角形→平行四辺形)
    if shear_range != 0:
        transform_list.append(torchvision.transforms.RandomAffine(0, shear=shear_range))

    # 拡大
    if zoom_range != 0:
        transform_list.append(torchvision.transforms.RandomAffine(0, scale=(1-zoom_range,1+zoom_range)))

    # 左右反転
    if horizontal_flip != 0:
        transform_list.append(torchvision.transforms.RandomHorizontalFlip())

    # 上下反転
    if vertical_flip != 0:
        transform_list.append(torchvision.transforms.RandomVerticalFlip())

    return torchvision.transforms.Compose(transform_list)

### Celebのimagepathリスト取得 ###
def makeImagePathList_Celeb(
        data_dir=os.getenv('FAKE_DATA_PATH'),
        classes=['Celeb-real-image-face-90', 'Celeb-synthesis-image-face-90'],
        train_rate=None,
        validation_rate=0.1,
        test_rate=0.1,
        data_type=None
    ):
    """
    CelebDFの画像パスとラベルの組み合わせのリストを取得
    """

    class_file_num = {}
    class_weights = {}
    train_data = []
    validation_data = []
    test_data = []
    if train_rate is None:
        train_rate = 1 - validation_rate - test_rate
        s1 = (int)(59*train_rate)
        s2 = (int)(59*(train_rate+validation_rate))
        s3 = (int)(59)
    elif (train_rate + validation_rate + test_rate) > 1:
        print(f"Cannot be split. (train_rate:{train_rate}, validation_rate:{validation_rate}, test_rate:{test_rate})")
        exit()
    else:
        s1 = (int)(59*train_rate)
        s2 = (int)(59*(train_rate+validation_rate))
        s3 = (int)(59*(train_rate+validation_rate+test_rate))
    id_list = list(range(62))
    id_list.remove(14)
    id_list.remove(15)
    id_list.remove(18)
    # random.shuffle(id_list)
    train_id_list = id_list[ : s1]
    validation_id_list = id_list[s1 : s2]
    test_id_list = id_list[s2 : s3]
    print("\tTRAIN IMAGE DATA ID: ",end="")
    print(train_id_list)
    print("\tVALIDATION IMAGE DATA ID: ",end="")
    print(validation_id_list)
    print("\tTEST IMAGE DATA ID: ",end="")
    print(test_id_list)
    del id_list
    data_num = 0
    for l,c in enumerate(classes):
        image_path_list = sorted(glob.glob(data_dir+"/"+c+"/*"))
        path_num = len(image_path_list)
        data_num += path_num
        regexp = r'^.+?id(?P<id>(\d+))(_id(?P<id2>\d+))?_(?P<key>\d+)_(?P<num>\d+).(?P<ext>.{2,4})$'
        past_path = image_path_list[0]
        movie_image_list = []
        for i in range(1,len(image_path_list)):
            past_ids = re.search(regexp,past_path).groupdict()
            now_ids = re.search(regexp,image_path_list[i]).groupdict()
            if (past_ids['id']==now_ids['id']) and (past_ids['id2']==None or past_ids['id2']==now_ids['id2']) and (past_ids['key']==now_ids['key']):
                movie_image_list.append([image_path_list[i],l])
            else:
                if int(past_ids['id']) in train_id_list:
                    train_data.append(movie_image_list)
                elif int(past_ids['id']) in validation_id_list:
                    validation_data.append(movie_image_list)
                elif int(past_ids['id']) in test_id_list:
                    test_data.append(movie_image_list)
                movie_image_list = []
                movie_image_list.append([image_path_list[i],l])
            past_path = image_path_list[i]
        # 不均衡データ調整
        class_file_num[c] = path_num
        if l==0:
            n = class_file_num[c]
        class_weights[l] = 1 / (class_file_num[c]/n)

    train_data = list(chain.from_iterable(train_data))
    validation_data = list(chain.from_iterable(validation_data))
    test_data = list(chain.from_iterable(test_data))
    if data_type=="train":
        return train_data
    elif data_type=="validation":
        return validation_data
    elif data_type=="test":
        return test_data
    else:
        return (train_data, validation_data, test_data, data_num, class_file_num, class_weights)

### 画像データセット取得 ###
class ImageDataset(torch.utils.data.Dataset):
    '''
    ファイルパスとラベルの組み合わせのリストからDatasetを作成

    Parameters
    ----------
    data: [[path,label],[path,label],....,[path,label]]
        パスをラベルのリスト
    image_size : tuple
        画像サイズ
    transform : torchvision.transforms
        transformオブジェクト

    Returns
    -------
    ImageDatasetインスタンス
    '''

    def __init__(self, data=None, num_classes=2, image_size=(3,256,256), transform=getTransforms()):
        self.transform = transform
        if len(image_size)==3:
            self.image_c = image_size[0]
            self.image_w = image_size[1]
            self.image_h = image_size[2]
        elif len(image_size)==2:
            self.image_c = -1
            self.image_w = image_size[0]
            self.image_h = image_size[1]
        else:
            raise Exception
        self.data = data
        self.data_num = len(data) if data!=None else 0
        self.num_classes = num_classes

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.transform(Image.open(self.data[idx][0]).resize((self.image_w,self.image_h)))
        out_label = torch.tensor(self.data[idx][1])
        # Image.fromarray((out_data * 255).to('cpu').detach().numpy().transpose(1, 2, 0).astype(np.uint8)).save("tmp.png")
        if self.num_classes==2:
            out_label = out_label.view(-1).float()
        else:
            out_label = F.one_hot(out_label,num_classes=self.num_classes).float()
        return out_data, out_label

### Celebデータ作成 ###
def getCelebDataLoader(
        batch_size=64,
        transform=None,
        data_dir=os.getenv('FAKE_DATA_PATH'),
        classes=['Celeb-real-image-face-90', 'Celeb-synthesis-image-face-90'],
        image_size=(3,256,256),
        train_rate=None,
        validation_rate=0,
        test_rate=0,
        shuffle=False
    ):

    # パス取得
    celeb_paths = makeImagePathList_Celeb(
        data_dir=data_dir,
        classes=classes,
        train_rate=train_rate,
        validation_rate=validation_rate,
        test_rate=test_rate,
        data_type='all'
    )
    train_paths = celeb_paths[0]
    validation_paths = celeb_paths[1]
    test_paths = celeb_paths[2]
    data_num = celeb_paths[3]
    class_file_num = celeb_paths[4]
    class_weights = celeb_paths[5]

    # Dataset作成 (validationとtestはtransformするべきか問題###要検討###)
    train_dataset = ImageDataset(data=train_paths,num_classes=len(classes),image_size=image_size,transform=transform)
    validation_dataset = ImageDataset(data=validation_paths,num_classes=len(classes),image_size=image_size,transform=transform)
    test_dataset = ImageDataset(data=test_paths,num_classes=len(classes),image_size=image_size,transform=transform)

    # DataLoader作成
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=torch.cuda.device_count(), pin_memory=True)
    if validation_rate > 0:
        # validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=torch.cuda.device_count(), pin_memory=True)
    else:
        validation_rate = None
    if test_rate > 0:
        # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=torch.cuda.device_count(), pin_memory=True)
    else:
        test_dataloader = None

    print("data num: "+str(data_num))
    print("class file num: "+str(class_file_num))
    print("class weights: "+str(class_weights))
    print("train data num: "+str(len(train_dataset)))
    print("validation data num: "+str(len(validation_dataset) if validation_rate>0 else 0))
    print("test data num: "+str(len(test_dataset) if test_rate>0 else 0))
    print("train data batch num: "+str(len(train_dataloader)))
    print("validation data batch num: "+str(len(validation_dataloader) if validation_rate>0 else 0))
    print("test data batch num: "+str(len(test_dataloader) if test_rate>0 else 0))

    return (train_dataloader, validation_dataloader, test_dataloader, data_num, class_file_num, class_weights)


### DFDCのimagepathリスト取得 ###
def makeImagePathList_Dfdc(
        data_dir=str(os.getenv('FAKE_DATA_PATH'))+"/dfdc-face-90",
        data_rate=1.0,
        data_type=None
    ):
    """
    DFDCの画像パスとラベルの組み合わせのリストを取得
    """

    train_0_data = []
    train_1_data = []
    valid_0_data = []
    valid_1_data = []
    test_0_data = []
    test_1_data = []
    train_ids = set()
    valid_ids = set()
    test_ids = set()
    regexp = r'^.+?\/?(?P<label>(\d+))_(?P<id>[a-zA-Z0-9]+)_(?P<num>\d+).(?P<ext>.{2,4})$'
    train_image_path_list = sorted(glob.glob(data_dir+"/train/*"))
    for path in train_image_path_list:
        info = re.search(regexp,path).groupdict()
        if int(info['label']) == 0:
            train_0_data.append( [[path,0]] )
        else:
            train_1_data.append( [[path,1]] )
        train_ids.add(info['id'])
    valid_image_path_list = sorted(glob.glob(data_dir+"/validation/*"))
    for path in valid_image_path_list:
        info = re.search(regexp,path).groupdict()
        if int(info['label']) == 0:
            valid_0_data.append( [[path,0]] )
        else:
            valid_1_data.append( [[path,1]] )
        valid_ids.add(info['id'])
    test_image_path_list = sorted(glob.glob(data_dir+"/test/*"))
    for path in test_image_path_list:
        info = re.search(regexp,path).groupdict()
        if int(info['label']) == 0:
            test_0_data.append( [[path,0]] )
        else:
            test_1_data.append( [[path,1]] )
        test_ids.add(info['id'])

    print(f"\tDFDC train movie num (all): {len(train_ids)}")
    print(f"\tDFDC validation movie num (all): {len(valid_ids)}")
    print(f"\tDFDC test movie num (all): {len(test_ids)}")
    print(f"\tDFDC train image num (all): {len(train_0_data)+len(train_1_data)}")
    print(f"\tDFDC validation image num (all): {len(valid_0_data)+len(valid_1_data)}")
    print(f"\tDFDC test image num (all): {len(test_0_data)+len(test_1_data)}")

    train_0_split = int(len(train_0_data) * data_rate)
    train_1_split = int(len(train_1_data) * data_rate)
    valid_0_split = int(len(valid_0_data) * data_rate)
    valid_1_split = int(len(valid_1_data) * data_rate)
    test_0_split = int(len(test_0_data) * data_rate)
    test_1_split = int(len(test_1_data) * data_rate)
    train_0_data = train_0_data[:train_0_split]
    train_1_data = train_1_data[:train_1_split]
    valid_0_data = valid_0_data[:valid_0_split]
    valid_1_data = valid_1_data[:valid_1_split]
    test_0_data = test_0_data[:test_0_split]
    test_1_data = test_1_data[:test_1_split]
    class_file_num = {0: len(train_0_data)+len(valid_0_data)+len(test_0_data), 1: len(train_1_data)+len(valid_1_data)+len(test_1_data)}
    train_class_weights = {0:len(train_1_data)/len(train_0_data), 1:len(train_0_data)/len(train_0_data)}
    train_data = train_0_data + train_1_data
    valid_data = valid_0_data + valid_1_data
    test_data = test_0_data + test_1_data

    print(f"\tDFDC train image num (use): {len(train_data)}")
    print(f"\tDFDC validation image num (use): {len(valid_data)}")
    print(f"\tDFDC test image num (use): {len(test_data)}")
    data_num = len(train_data) + len(valid_data) + len(test_data)

    train_data = list(chain.from_iterable(train_data))
    valid_data = list(chain.from_iterable(valid_data))
    test_data = list(chain.from_iterable(test_data))
    if data_type=="train":
        return train_data
    elif data_type=="validation":
        return valid_data
    elif data_type=="test":
        return test_data
    else:
        return (train_data, valid_data, test_data, data_num, class_file_num, train_class_weights)

### DFDCデータ作成 ###
def getDfdcDataLoader(
        batch_size=64,
        transform=None,
        data_dir=str(os.getenv('FAKE_DATA_PATH'))+"/dfdc-face-90",
        image_size=(3,256,256),
        data_rate=1.0,
        shuffle=False
    ):

    # パス取得
    celeb_paths = makeImagePathList_Dfdc(
        data_dir=data_dir,
        data_rate=data_rate,
        data_type='all'
    )
    train_paths = celeb_paths[0]
    validation_paths = celeb_paths[1]
    test_paths = celeb_paths[2]
    data_num = celeb_paths[3]
    class_file_num = celeb_paths[4]
    class_weights = celeb_paths[5]

    # Dataset作成 (validationとtestはtransformするべきか問題###要検討###)
    train_dataset = ImageDataset(data=train_paths,num_classes=2,image_size=image_size,transform=transform)
    validation_dataset = ImageDataset(data=validation_paths,num_classes=2,image_size=image_size,transform=transform)
    test_dataset = ImageDataset(data=test_paths,num_classes=2,image_size=image_size,transform=transform)

    # DataLoader作成
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
    # validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

    print("data num: "+str(data_num))
    print("class file num: "+str(class_file_num))
    print("class weights: "+str(class_weights))
    print("train data batch num: "+str(len(train_dataloader)))
    print("validation data batch num: "+str(len(validation_dataloader)))
    print("test data batch num: "+str(len(test_dataloader)))

    return (train_dataloader, validation_dataloader, test_dataloader, data_num, class_file_num, class_weights)


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
