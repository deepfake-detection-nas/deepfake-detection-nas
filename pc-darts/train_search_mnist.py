import os
import sys
import time
import datetime
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

# from torch.autograd import Variable
from model_search_celeb import Network, counter_input
from architect_celeb import Architect
from common_import import *


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes in your custom dataset')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='MNIST', help='experiment name')
parser.add_argument('--restart', type=str, default=None, help='restart directory')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

image_size = (1,28,28)
validation_rate = 0.15
test_rate = 0.7
normalization = False
rotation_range=15.0
width_shift_range=0.15
height_shift_range=0.15
brightness_range = 0.0
shear_range=0.0
zoom_range=0.1
horizontal_flip=True
vertical_flip=False
# validation_rate = 0.018
# test_rate = 0.964
# normalization = False
# rotation_range=0.0
# width_shift_range=0.0
# height_shift_range=0.0
# brightness_range = 0.0
# shear_range=0.0
# zoom_range=0.0
# horizontal_flip=False
# vertical_flip=False

args.image_size = image_size
args.validation_rate = validation_rate
args.test_rate = test_rate
args.normalization = normalization
args.rotation_range = rotation_range
args.width_shift_range = width_shift_range
args.height_shift_range = height_shift_range
args.brightness_range = brightness_range
args.shear_range = shear_range
args.zoom_range = zoom_range
args.horizontal_flip = horizontal_flip
args.vertical_flip = vertical_flip

if args.restart is None:
  args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
  args.restart = args.save
  saveArgs(args, os.path.join(args.save, 'param.json'))
  args.restart = None
else:
  args.save = args.restart
  print('Experiment dir : {}'.format(args.save))
  loadArgs(args, os.path.join(args.save, 'param.json'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.restart is not None:
  checkpoint_path = os.path.join(args.save, 'state.pt')
  if not os.path.exists(checkpoint_path):
    logging.info('Cannot restart because not exist checkpoint.')
    sys.exit(1)
  logging.info('')
  logging.info('Resuming from checkpoint...')

device = torch.device("cuda")
args.batch_size = args.batch_size * torch.cuda.device_count()
args.image_size = tuple(args.image_size)


def main(train_queue, valid_queue, pos_weight=1):

  np.random.seed(args.seed)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.to(device)
  model = Network(args.init_channels, args.num_classes, args.layers, criterion, gray=True)
  model = model.to(device)
  logging.info("model size = %fMB", get_model_memory_size(model)/1024)
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)
  for state in optimizer.state.values():
    for k, v in state.items():
      if isinstance(v, torch.Tensor):
        state[k] = v.to(device)

  # num_train = len(train_data)
  # indices = list(range(num_train))
  # split = int(np.floor(args.train_portion * num_train))

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  # 再開であれば読み込み
  if args.restart is not None:
    model, optimizer, scheduler, architect_optim, start_epoch, start_step = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
  else:
    start_epoch = 0
    start_step = 0
  logging.info("saved epoch: "+str(start_epoch))
  logging.info("saved step: "+str(start_step))

  # Multi GPU使用宣言
  if torch.cuda.device_count() > 0:
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
    logging.info("Multi GPU OK.")

  if args.restart is not None:
    architect = Architect(model, criterion, args, architect_optim)
  else:
    architect = Architect(model, criterion, args)

  # torchsummary.summary(model, args.image_size, device='cuda')

  for epoch in range(start_epoch+1, args.epochs+1):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.module.genotype()
    logging.info('genotype = %s', genotype)

    #print(F.softmax(model.alphas_normal, dim=-1))
    #print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    logging.info(f"Epoch: {epoch}")
    start_step = start_step if epoch == start_epoch+1 else 0
    train_acc, train_obj, t_epoch = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, scheduler, start_step)
    logging.info('train_acc %f', train_acc)
    logging.info('epoch_train_time %s', t_epoch)

    # validation
    # if args.epochs-epoch<=0: # ←1だった
    #   valid_acc, valid_obj = infer(valid_queue, model, criterion)
    #   logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))
    state = {
      'model': model.module.state_dict(),
      'optimizer': optimizer.state_dict(),
      'scheduler': scheduler.state_dict(),
      'architect_optim': architect.optimizer.state_dict(),
      'epoch': epoch,
      'step': 0,
      'loss': train_obj
    }
    save_checkpoint(state, os.path.join(args.save, 'state.pt'))

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, scheduler, start_step=0):
  objs = utils.AvgrageMeter()
  obja = utils.AvgrageMeter()

  t_sum = 0
  bar_length = 20
  checkpoint_interval = math.floor(len(train_queue) / 5)
  for step, (input, target) in enumerate(train_queue,start=1):
    if step < start_step+1:
      continue
    t = time.time()
    model.train()
    n = input.size(0)

    input = input.to(device)
    target = target.to(device)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    #try:
    #  input_search, target_search = next(valid_queue_iter)
    #except:
    #  valid_queue_iter = iter(valid_queue)
    #  input_search, target_search = next(valid_queue_iter)
    input_search = input_search.to(device)
    target_search = target_search.to(device)

    if epoch>=15:
      architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(model.module.parameters(), args.grad_clip)
    optimizer.step()

    acc = accuracy(logits, target)
    objs.update(loss.item(), n)
    obja.update(acc.item(), n)

    # if step % args.report_freq == 0:
    #   logging.info('train %03d %e %f', step, objs.avg, obja.avg)

    if step % checkpoint_interval == 0:
      state = {
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'architect_optim': architect.optimizer.state_dict(),
        'epoch': epoch-1,
        'step': step,
        'loss': objs.avg
      }
      save_checkpoint(state, os.path.join(args.save, 'state.pt'))

    interval = time.time() - t
    t_sum += interval
    eta = str(datetime.timedelta( seconds=int((len(train_queue)-step)*interval) ))
    done = math.floor(bar_length * step / len(train_queue))
    bar = '='*done + ("=" if done == bar_length else ">")  + "."*(bar_length-done)
    print_log(f"\r  \033[K[{bar}] - ETA: {eta:>8}, {math.floor(step / len(train_queue)*100):>3}% ({step}/{len(train_queue)})", line_break=False)
    print_log(f", Loss: {objs.avg:.04f}", line_break=False)
    print_log(f", Acc: {obja.avg:.04f}", line_break=False)
    print_log(f"\t", line_break=False)
  t_total = str(datetime.timedelta( seconds=int(t_sum) ))
  bar = '='*bar_length
  print_log(f"\r  \033[K[{bar}] - {t_total:>8}, 100% ({len(train_queue)}/{len(train_queue)})", line_break=False)
  print_log(f", Loss: {objs.avg:.04f}", line_break=False)
  print_log(f", Acc: {obja.avg:.04f}", line_break=False)
  print_log(f"\t", line_break=False)
  print_log()

  return obja.avg, objs.avg, t_total


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  obja = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    #input = input.cuda()
    #target = target.cuda(non_blocking=True)
    input = input.to(device)
    target = target.to(device)
    logits = model(input)
    loss = criterion(logits, target)

    acc = accuracy(logits, target)
    n = input.size(0)
    objs.update(loss.item(), n)
    obja.update(acc.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f', step, objs.avg, obja.avg)

  return obja.avg, objs.avg


if __name__ == '__main__':
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  transform = getTransforms(
    image_size=args.image_size,
    normalization=args.normalization,
    rotation_range=args.rotation_range,
    width_shift_range=args.width_shift_range,
    height_shift_range=args.height_shift_range,
    brightness_range=args.brightness_range,
    shear_range=args.shear_range,
    zoom_range=args.zoom_range,
    horizontal_flip=args.horizontal_flip,
    vertical_flip=args.vertical_flip
  )
  from torchvision import datasets, transforms
  from torch.utils.data import DataLoader
  train_data = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
  train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
  test_data = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
  test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
  print("GPU: " + str(torch.cuda.is_available()))
  main(train_loader, test_loader)

  # 終了証明
  f = open(os.path.join(args.save, 'finish_training'), 'w')
  f.write('')  # 何も書き込まなくてファイルは作成されました
  f.close()
