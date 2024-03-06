import glob
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torchvision.datasets as dset

import utils as utils
from model_search_celeb import Network, counter_input
from architect_celeb import Architect
from search_config import args

parent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parent_folder_path)
from common_import_darts import *





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


def main(train_queue, valid_queue, test_queue, pos_weight=1):
    utils.set_seed(seed=args.seed)
    logging.info("args = %s", args)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float).to(device))
    criterion = criterion.to(device)
    model = Network(args, args.init_channels, args.num_classes, args.layers, criterion)
    model = model.to(device)
    metrics_dict = getMetrics(device, mode="all", num_classes=args.num_classes, average='none')
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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

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

    torchsummary.summary(model, args.image_size, device='cuda')

    # if args.debug:
    #     split = args.batch_size
    #     num_train = 2 * args.batch_size

    for epoch in range(start_epoch+1, args.epochs+1):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        # training
        logging.info(f"Epoch: {epoch}")
        start_step = start_step if epoch == start_epoch+1 else 0
        train_acc, train_obj, train_history = train(args, logging, train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, scheduler, start_step, metrics_dict=metrics_dict)
        logging.info('train_acc %f', train_acc)
        for key,value in train_history.items():
            logging.info(f'train_{key} {value}')
        # logging.info('epoch_train_time %s', train_history['elapsed_time'])

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

        torch.cuda.empty_cache()

        # validation
        # valid_acc, valid_obj, valid_history = infer(args, logging, valid_queue, model, criterion, metrics_dict=metrics_dict)
        # logging.info('valid_acc %f', valid_acc)
        # for key,value in valid_history.items():
        #     logging.info(f'valid_{key} {value}')
        # torch.cuda.empty_cache()

        # print arch params
        # logging.info('normal alpha = %s', F.softmax(model.module.alphas_normal, dim=-1))
        # logging.info('normal beta = %s', F.sigmoid(model.module.betas_normal))
        # logging.info('reduce alpha = %s', F.softmax(model.module.alphas_reduce, dim=-1))
        # logging.info('reduce beta = %s', F.sigmoid(model.module.betas_reduce))

        # genotype = model.module.genotype()
        # logging.info('genotype = %s', genotype)

    logging.info('\n\nTest:')
    torch.cuda.empty_cache()
    test_acc, test_obj, test_history = infer(args, logging, test_queue, model, criterion, metrics_dict=metrics_dict)
    logging.info('test_acc (last test) %f', test_acc)
    for key,value in test_history.items():
        logging.info(f'test_{key} (last test) {value}')
    torch.cuda.empty_cache()


if __name__ == '__main__':
    # utils.run_func(args, main)
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
    train_dataloader, validation_dataloader, test_dataloader, \
    data_num, class_file_num, class_weights = getCelebDataLoader(
        batch_size=args.batch_size,
        transform=transform,
        data_dir=os.getenv('FAKE_DATA_PATH'),
        image_size=args.image_size,
        train_rate=args.train_rate,
        validation_rate=args.validation_rate,
        test_rate=args.test_rate,
        use_data_rate=args.use_data_rate,
        seed=args.seed,
        shuffle=True,
        num_workers=0
    )
    print("GPU: " + str(torch.cuda.is_available()))
    main(train_dataloader, validation_dataloader, test_dataloader, class_weights[1])

    # 終了証明
    f = open(os.path.join(args.save, 'finish_training'), 'w')
    f.write('')  # 何も書き込まなくてファイルは作成されました
    f.close()
