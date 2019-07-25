#coding=utf-8
from __future__ import print_function
import torch
import torchvision.transforms as transforms

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil.tz
import time
from miscc.config import cfg, cfg_from_file

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.'))) #获取当前执行脚本的绝对路�?sys.path.append(dir_path)

def parse_args():#parameter setting /input sample:python main.py --cfg cfg/birds_3stages.yml --gpu 0
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/birds_3stages.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != '-1':
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg) #pprint()模块打印出来的数据结构更加完�?每行为一个数据结�?更加方便阅读打印输出结果

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed) #关闭进程,重新运行上面代码,发现a=torch.rand([1,5]) 得到的是一样的
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed) #为所有的GPU设置种子
    now = datetime.datetime.now(dateutil.tz.tzlocal()) #acquire current time
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    #output_dir = '../output/%s_%s_%s' % \
    #    (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp) #'birds'/'3stages'/2018-07-02-11-17-33
    output_dir = '../output/256_birds_2021'

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG: #True
        if cfg.DATASET_NAME == 'birds':
            bshuffle = False
            split_dir = 'test'
        else:
            bshuffle = False
            split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1)) #branch_num=3 BASE_SIZE=64
    image_transform = transforms.Compose([       #torchvision.transforms是pytorch中的图像预处理包
        transforms.Scale(int(imsize * 76 / 64)), #2d平面上调整元素大小的变换 transform=transforms.Compose
        transforms.RandomCrop(imsize),           #依据给定的size随机裁剪([transforms.ToTensor(),
        transforms.RandomHorizontalFlip()])      #依据概率p对PIL图片进行水平翻转 默认值为0.5 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    if cfg.GAN.B_CONDITION:  # text to image task
        from datasets import TextDataset
        dataset = TextDataset(cfg.DATA_DIR, split_dir,
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
    assert dataset
    num_gpu = len(cfg.GPU_ID.split(','))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))
    print('dataloader:',dataloader,type(dataloader))

    # Define models and go to train/evaluate
    if not cfg.GAN.B_CONDITION:  #default=True
        from trainer00 import GANTrainer as trainer
    else:
        from trainer00 import condGANTrainer as trainer #condition
    algo = trainer(output_dir, dataloader, imsize)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        algo.evaluate(split_dir)
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
