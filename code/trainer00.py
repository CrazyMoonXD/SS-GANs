#coding=utf-8
from __future__ import print_function
from six.moves import range
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch import autograd
import torchvision.utils as vutils
import numpy as np
import os
import time
from PIL import Image
from copy import deepcopy
from miscc.config import cfg
from miscc.utils import mkdir_p
import cPickle as pkl
from model import G_NET_64, G_NET_128, G_NET_256, D_NET64, D_NET128, D_NET256, INCEPTION_V3

# ################## Shared functions ###################
def compute_mean_covariance(img):
    batch_size = img.size(0)
    channel_num = img.size(1)
    height = img.size(2)
    width = img.size(3)
    num_pixels = height * width
    # batch_size * channel_num * 1 * 1
    mu = img.mean(2, keepdims=True).mean(3, keepdims=True)
    # batch_size * channel_num * num_pixels
    img_hat = img - mu.expand_as(img)
    img_hat = img_hat.view(batch_size, channel_num, num_pixels) #batch_size * channel_num * num_pixels
    img_hat_transpose = img_hat.transpose(1, 2)  #transpose: batch_size * num_pixels * channel_num
    covariance = torch.bmm(img_hat, img_hat_transpose) #矩阵乘法 covariance: batch_size * channel_num * channel_num
    covariance = covariance / num_pixels #height * width
    return mu, covariance

def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar) #logvar
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param): #该函数返回一个以元组为元素的列表，其中第 i 个元组包含每个参数序列的�?i 个元�?
        p.data.copy_(new_p) #new_p=new_param

def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters())) #list[p.data1,p.data2,...,p.data]
    return flatten

def compute_inception_score(predictions, num_splits=1):
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        kl = part * \
            (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

def negative_log_posterior_probability(predictions, num_splits=1):
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        result = -1. * np.log(np.max(part, 1))
        result = np.mean(result)
        scores.append(result)
    return np.mean(scores), np.std(scores)

def load_network(gpus):
    netG_64 = G_NET_64()
    netG_128 = G_NET_128()
    netG_256 = G_NET_256()
    netG_64.apply(weights_init)
    netG_128.apply(weights_init)
    netG_256.apply(weights_init)
    netG_64 = torch.nn.DataParallel(netG_64, device_ids=gpus)
    netG_128 = torch.nn.DataParallel(netG_128, device_ids=gpus)
    netG_256 = torch.nn.DataParallel(netG_256, device_ids=gpus)
    print(netG_256)
    netsD = []
    if cfg.TREE.BRANCH_NUM > 0:
        netsD.append(D_NET64())
    if cfg.TREE.BRANCH_NUM > 1:
        netsD.append(D_NET128())
    if cfg.TREE.BRANCH_NUM > 2:
        netsD.append(D_NET256())
    for i in range(len(netsD)):
        netsD[i].apply(weights_init)
        netsD[i] = torch.nn.DataParallel(netsD[i], device_ids=gpus)  #multi GPU setting
    print('# of netsD', len(netsD))
    count = 0
    if cfg.TRAIN.NET_G_64 != '':
        state_dict = torch.load(cfg.TRAIN.NET_G_64)   #load G network
        netG_64.load_state_dict(state_dict)   #load model recommand
        print('Load ', cfg.TRAIN.NET_G_64)    #visualize network
    if cfg.TRAIN.NET_G_128 != '':
        state_dict = torch.load(cfg.TRAIN.NET_G_128)  # load G network
        netG_128.load_state_dict(state_dict)  # load model recommand
        print('Load ', cfg.TRAIN.NET_G_128)  # visualize network
        #istart = cfg.TRAIN.NET_G.rfind('_') + 1   #字符串最后一次出现的位置(从右向左查询)，如果没有匹配项
        #iend = cfg.TRAIN.NET_G.rfind('.')
        #count = cfg.TRAIN.NET_G[istart:iend]   ######## netG_2000.pth
        #count = int(count) + 1
    if cfg.TRAIN.NET_D != '':
        for i in range(len(netsD)):
            print('Load %s_%d.pth' % (cfg.TRAIN.NET_D, i))
            state_dict = torch.load('%s%d.pth' % (cfg.TRAIN.NET_D, i))
            netsD[i].load_state_dict(state_dict)
    inception_model = INCEPTION_V3()
    if cfg.CUDA:
        netG_64.cuda()
        netG_128.cuda()
        netG_256.cuda()
        for i in range(len(netsD)):
            netsD[i].cuda()
        inception_model = inception_model.cuda()
    inception_model.eval()
    return netG_64, netsD, len(netsD), inception_model, count, netG_128 ,netG_256

def define_optimizers(netG_64,netG_128,netG_256,netsD):
    optimizersD = []
    num_Ds = len(netsD)
    for i in range(num_Ds):
        opt = optim.Adam(netsD[i].parameters(),
                         lr=cfg.TRAIN.DISCRIMINATOR_LR, #learning rate=0.0002
                         betas=(0.5, 0.999))
        optimizersD.append(opt)
    #print('#################parameters:',type(netG_64.parameters()))
    optimizerG_64 = optim.Adam(netG_64.parameters(),
                            lr=cfg.TRAIN.GENERATOR_LR, #learning rate=0.0002
                            betas=(0.5, 0.999))
    optimizerG_128 = optim.Adam(netG_128.parameters(),
                            lr=cfg.TRAIN.GENERATOR_LR, #learning rate=0.0002
                            betas=(0.5, 0.999))
    optimizerG_256 = optim.Adam(netG_256.parameters(),
                            lr=cfg.TRAIN.GENERATOR_LR, #learning rate=0.0002
                            betas=(0.5, 0.999))
    return optimizerG_64, optimizerG_128, optimizerG_256, optimizersD

def save_model(netG_64, avg_param_G_64,netG_128, avg_param_G_128,netG_256, avg_param_G_256,netsD,epoch, model_dir):
    load_params(netG_64, avg_param_G_64)
    torch.save(
        netG_64.state_dict(),
        '%s/netG_64_%d.pth' % (model_dir, epoch)) #save G model
    load_params(netG_128, avg_param_G_128)
    torch.save(
        netG_128.state_dict(),
        '%s/netG_128_%d.pth' % (model_dir, epoch)) #save G model
    load_params(netG_256, avg_param_G_256)
    torch.save(
        netG_256.state_dict(),
        '%s/netG_256_%d.pth' % (model_dir, epoch)) #save G model
    for i in range(len(netsD)):
        netD = netsD[i]
        torch.save(
            netD.state_dict(),          #save D model
            '%s/netD%d.pth' % (model_dir, i))
    print('Save G/Ds models.')

def save_img_results(imgs_tcpu, fake_imgs_64,fake_imgs_128,fake_imgs_256,num_imgs,
                     count, image_dir):
    num = cfg.TRAIN.VIS_COUNT  #TRAIN.VIS_COUNT = 64
    real_img = imgs_tcpu[-1][0:num] #the range of real_img
    vutils.save_image(real_img, '%s/real_samples.png' % (image_dir), normalize=True) #change to [0,1]
    real_img_set = vutils.make_grid(real_img).numpy()  #grid
    real_img_set = np.transpose(real_img_set, (1, 2, 0))
    real_img_set = real_img_set * 255
    real_img_set = real_img_set.astype(np.uint8)
    for i in range(num_imgs):
        if i == 0:
            fake_img = fake_imgs_64[0][0:num] #num_imgs=0,1,2 num=24
        elif i == 1:
            fake_img = fake_imgs_128[0][0:num]  # num_imgs=0,1,2 num=24
        elif i == 2:
            fake_img = fake_imgs_256[0][0:num]  # num_imgs=0,1,2 num=24
        vutils.save_image(fake_img.data, '%s/count_%09d_fake_samples%d.png'
                          %(image_dir, count, i), normalize=True)  #is still [-1. 1]
        fake_img_set = vutils.make_grid(fake_img.data).cpu().numpy()
        fake_img_set = np.transpose(fake_img_set, (1, 2, 0))
        fake_img_set = (fake_img_set + 1) * 255 / 2
        fake_img_set = fake_img_set.astype(np.uint8)

# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, imsize):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True #如果网络的输入数据维度或类型上变化不�?自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL  #snapshot=2000
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

    def prepare_data(self, data):
        imgs, w_imgs, t_embedding, _,wrong_label,real_label = data
        real_vimgs, wrong_vimgs = [], []
        if cfg.CUDA:
            vembedding = Variable(t_embedding).cuda()
            wrong_label = Variable(wrong_label).cuda()
            real_label = Variable(real_label).cuda()
        else:
            vembedding = Variable(t_embedding)
            wrong_label = Variable(wrong_label)
            real_label = Variable(real_label)
        for i in range(self.num_Ds):
            if cfg.CUDA:
                real_vimgs.append(Variable(imgs[i]).cuda())
                wrong_vimgs.append(Variable(w_imgs[i]).cuda())
            else:
                real_vimgs.append(Variable(imgs[i]))
                wrong_vimgs.append(Variable(w_imgs[i]))
        return imgs, real_vimgs, wrong_vimgs, vembedding, wrong_label,real_label

    def train_Dnet(self, idx, count):
        batch_size = self.real_imgs[0].size(0)
        criterion, mu = self.criterion, self.mu
        netD, optD = self.netsD[idx], self.optimizersD[idx]
        real_imgs = self.real_imgs[idx]
        wrong_imgs = self.wrong_imgs[idx]
        loss_label = nn.CrossEntropyLoss()
        l2_distance = nn.MSELoss()
        if idx == 0:
            fake_imgs = self.fake_imgs_64
        elif idx == 1:
            fake_imgs = self.fake_imgs_128
        elif idx == 2:
            fake_imgs = self.fake_imgs_256
        netD.zero_grad()

        real_labels = self.real_labels[:batch_size]
        fake_labels = self.fake_labels[:batch_size]
        real_label = self.real_image_label
        wrong_label = self.wrong_image_label

        real_logits = netD(real_imgs, mu.detach())
        wrong_logits = netD(wrong_imgs, mu.detach())
        fake_logits = netD(fake_imgs[0].detach(), mu.detach())

        errD_real = criterion(real_logits[0], real_labels)
        errD_wrong = criterion(wrong_logits[0], fake_labels)
        errD_fake = criterion(fake_logits[0], fake_labels)

        real = real_logits[2].squeeze()
        fake = fake_logits[2].squeeze()
        wrong = wrong_logits[2].squeeze()

        first_part = loss_label(real, real_label)
        second_part = loss_label(fake, real_label)
        third_part = loss_label(wrong, wrong_label)
        loss_tac = first_part + second_part + third_part
        if idx == 2:
            lsgan_loss = 0.5 * (l2_distance(real_logits[0], real_labels) + \
            l2_distance(fake_logits[0], fake_labels))
        else:
            lsgan_loss = 0

        if len(real_logits) > 1 and cfg.TRAIN.COEFF.UNCOND_LOSS > 0:
            errD_real_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
                criterion(real_logits[1], real_labels)
            errD_wrong_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
                criterion(wrong_logits[1], real_labels)
            errD_fake_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
                criterion(fake_logits[1], fake_labels)
            errD_real = errD_real + errD_real_uncond
            errD_wrong = errD_wrong + errD_wrong_uncond
            errD_fake = errD_fake + errD_fake_uncond

            errD = errD_real + 0.5 * (errD_wrong + errD_fake) + loss_tac + lsgan_loss #tac+wgan_gp
        else:
            errD = errD_real + 0.5 * (errD_wrong + errD_fake)
        errD.backward()
        optD.step()
        return errD

    def train_Gnet(self, count):
        self.netG_64.zero_grad()
        self.netG_128.zero_grad()
        self.netG_256.zero_grad()
        errG_total = 0
        flag = count % 100
        loss_label = nn.CrossEntropyLoss()
        batch_size = self.real_imgs[0].size(0)
        criterion, mu, logvar = self.criterion, self.mu, self.logvar
        real_labels = self.real_labels[:batch_size]
        l2_distance = nn.MSELoss()
        for i in range(self.num_Ds):
            if i == 0:
                outputs = self.netsD[i](self.fake_imgs_64[0], mu)
                lsgan_loss = 0
            elif i == 1:
                outputs = self.netsD[i](self.fake_imgs_128[0], mu)
                lsgan_loss = 0
            elif i == 2:
                outputs = self.netsD[i](self.fake_imgs_256[0], mu)
                lsgan_loss = l2_distance(outputs[0], real_labels)
            real_label = self.real_image_label
            errG = criterion(outputs[0], real_labels)
            output_value = outputs[2].squeeze()
            second_part = loss_label(output_value, real_label)
            if len(outputs) > 1 and cfg.TRAIN.COEFF.UNCOND_LOSS > 0:
                errG_patch = cfg.TRAIN.COEFF.UNCOND_LOSS *\
                    criterion(outputs[1], real_labels)
                errG = errG + errG_patch   #wgan+gan
            errG_total = errG_total + errG + second_part + lsgan_loss
        if cfg.TRAIN.COEFF.COLOR_LOSS > 0: # Compute color consistency losses conditional situation is about 0 useless
            if self.num_Ds > 1:
                mu1, covariance1 = compute_mean_covariance(self.fake_imgs[-1])
                mu2, covariance2 = compute_mean_covariance(self.fake_imgs[-2].detach())
                like_mu2 = cfg.TRAIN.COEFF.COLOR_LOSS * nn.MSELoss()(mu1, mu2)
                like_cov2 = cfg.TRAIN.COEFF.COLOR_LOSS * 5 * nn.MSELoss()(covariance1, covariance2)
                errG_total = errG_total + like_mu2 + like_cov2
            if self.num_Ds > 2:
                mu1, covariance1 = compute_mean_covariance(self.fake_imgs[-2])
                mu2, covariance2 = compute_mean_covariance(self.fake_imgs[-3].detach())
                like_mu1 = cfg.TRAIN.COEFF.COLOR_LOSS * nn.MSELoss()(mu1, mu2)
                like_cov1 = cfg.TRAIN.COEFF.COLOR_LOSS * 5 * nn.MSELoss()(covariance1, covariance2)
                errG_total = errG_total + like_mu1 + like_cov1
        kl_loss = KL_loss(mu, logvar) * cfg.TRAIN.COEFF.KL
        errG_total = errG_total + kl_loss  #gan/wgan-gp
        errG_total.backward()
        self.optimizerG_64.step()
        self.optimizerG_128.step()
        self.optimizerG_256.step()
        return kl_loss, errG_total

    def train(self):
        self.netG_64, self.netsD, self.num_Ds, \
            self.inception_model, start_count,self.netG_128,self.netG_256 = load_network(self.gpus)
        avg_param_G_64 = copy_G_params(self.netG_64)
        avg_param_G_128 = copy_G_params(self.netG_128)
        avg_param_G_256 = copy_G_params(self.netG_256)
        self.optimizerG_64,self.optimizerG_128,self.optimizerG_256,self.optimizersD = \
            define_optimizers(self.netG_64,self.netG_128,self.netG_256,self.netsD)
        self.criterion = nn.BCELoss()  #Binary_Corss_Entropy 单目标二分类交叉熵函
        self.real_labels = Variable(torch.FloatTensor(self.batch_size).fill_(1))
        self.fake_labels = Variable(torch.FloatTensor(self.batch_size).fill_(0))
        self.gradient_one = torch.FloatTensor([1.0])
        self.gradient_half = torch.FloatTensor([0.5])
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(self.batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(self.batch_size, nz).normal_(0, 1))
        if cfg.CUDA:
            self.criterion.cuda()
            self.real_labels = self.real_labels.cuda()
            self.fake_labels = self.fake_labels.cuda()
            self.gradient_one = self.gradient_one.cuda()
            self.gradient_half = self.gradient_half.cuda()
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
        predictions = []
        count = start_count
        start_epoch = start_count // (self.num_batches)
        loss = {}
        inception_number = 0
        inception = {}
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()
            print(type(self.data_loader))
            for step, data in enumerate(self.data_loader, 0):
                self.imgs_tcpu, self.real_imgs, self.wrong_imgs, \
                    self.txt_embedding,self.wrong_image_label,self.real_image_label = self.prepare_data(data)
                self.wrong_image_label = self.wrong_image_label.long()
                self.real_image_label = self.real_image_label.long()
                noise.data.normal_(0, 1)
                self.fake_imgs_64, _, _, h_code_64 = \
                    self.netG_64(noise, self.txt_embedding)
                self.fake_imgs_128, _, _, h_code_128 = \
                    self.netG_128(noise, h_code_64, self.txt_embedding)
                self.fake_imgs_256, self.mu, self.logvar,_ = \
                    self.netG_256(noise, h_code_128, self.txt_embedding)
                errD_total = 0
                for i in range(self.num_Ds):
                    errD = self.train_Dnet(i, count)
                    errD_total += errD
                kl_loss, errG_total = self.train_Gnet(count)
                for p_64, avg_p_64 in zip(self.netG_64.parameters(), avg_param_G_64):
                    avg_p_64.mul_(0.999).add_(0.001, p_64.data)
                for p_128, avg_p_128 in zip(self.netG_128.parameters(), avg_param_G_128):
                    avg_p_128.mul_(0.999).add_(0.001, p_128.data)
                for p_256, avg_p_256 in zip(self.netG_256.parameters(), avg_param_G_256):
                    avg_p_256.mul_(0.999).add_(0.001, p_256.data)
                # for inception score
                pred = self.inception_model(self.fake_imgs_64[0].detach())
                predictions.append(pred.data.cpu().numpy())
                count = count + 1
                if count % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                    save_model(self.netG_64, avg_param_G_64,self.netG_128, avg_param_G_128,self.netG_256, avg_param_G_256,self.netsD, count, self.model_dir)
                    backup_para_64 = copy_G_params(self.netG_64)
                    load_params(self.netG_64, avg_param_G_64)
                    backup_para_128 = copy_G_params(self.netG_128)
                    load_params(self.netG_128, avg_param_G_128)
                    backup_para_256 = copy_G_params(self.netG_256)
                    load_params(self.netG_256, avg_param_G_256)
                    self.fake_imgs_64, _, _,self.h_code_64 = \
                        self.netG_64(fixed_noise, self.txt_embedding)
                    self.fake_imgs_128, _, _, self.h_code_128= \
                        self.netG_128(fixed_noise, self.h_code_64,self.txt_embedding)
                    self.fake_imgs_256, _, _, self.h_code_256= \
                        self.netG_256(fixed_noise, self.h_code_128,self.txt_embedding)
                    save_img_results(self.imgs_tcpu, self.fake_imgs_64,self.fake_imgs_128,self.fake_imgs_256,self.num_Ds,
                                     count, self.image_dir)
                    load_params(self.netG_64, backup_para_64)
                    load_params(self.netG_128, backup_para_128)
                    load_params(self.netG_256, backup_para_256)
                    # Compute inception score
                    if len(predictions) > 500:
                        inception_number = inception_number + 1
                        inception[inception_number] = []
                        predictions = np.concatenate(predictions, 0)
                        mean, std = compute_inception_score(predictions, 10)
                        mean_nlpp, std_nlpp = \
                            negative_log_posterior_probability(predictions, 10)
                        #print('#########mean_nlpp:',mean_nlpp)
                        predictions = []
                        inception[inception_number].append(mean)
                        inception[inception_number].append(std)
                        inception[inception_number].append(mean_nlpp)
                        inception[inception_number].append(std_nlpp)
                    with open('loss_256_flowers_tac+h.pkl', 'wb') as f:
                        pkl.dump(loss, f)
                    with open('inception_256_flowers_tac+h.pkl', 'wb') as f:
                        pkl.dump(inception, f)
            end_t = time.time()
            print('''[%d/%d][%d]
                         Loss_D: %.2f Loss_G: %.2f Loss_KL: %.2f Time: %.2fs
                      '''
                  % (epoch, self.max_epoch, self.num_batches,           #result of training
                     errD_total.item(), errG_total.item(),
                     kl_loss.item(), end_t - start_t))
            D_loss = []
            G_loss = []
            KL_loss = []
            D_loss.append(errD_total.item())
            G_loss.append(errG_total.item())
            KL_loss.append(kl_loss.item())
            loss[epoch] = {}
            loss[epoch]['D_loss'] = D_loss
            loss[epoch]['G_loss'] = G_loss
            loss[epoch]['kl_loss'] = KL_loss
        save_model(self.netG_64, avg_param_G_64,self.netG_128, avg_param_G_128,self.netG_256, avg_param_G_256,self.netsD, count, self.model_dir)

    def save_superimages(self, images_list, filenames,
                         save_dir, split_dir, imsize):
        batch_size = images_list[0].size(0)
        num_sentences = len(images_list)
        for i in range(batch_size):
            s_tmp = '%s/super/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            savename = '%s_%d.png' % (s_tmp, imsize)
            super_img = []
            for j in range(num_sentences):
                img = images_list[j][i]
                img = img.view(1, 3, imsize, imsize)
                super_img.append(img)
            super_img = torch.cat(super_img, 0)  #
            vutils.save_image(super_img, savename, nrow=10, normalize=True)

    def save_singleimages(self, images, filenames,
                          save_dir, split_dir, sentenceID, imsize):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            fullpath = '%s_%d_sentence%d.png' % (s_tmp, imsize, sentenceID)
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()   # range from [-1, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def evaluate(self, split_dir):
        if cfg.TRAIN.NET_G_64 == '':
            print('Error: the path for morels is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            netG_64 = G_NET_64()
            netG_128 = G_NET_128()
            netG_256 = G_NET_256()
            netG_64.apply(weights_init)
            netG_128.apply(weights_init)
            netG_256.apply(weights_init)
            netG_64 = torch.nn.DataParallel(netG_64, device_ids=self.gpus)
            netG_128 = torch.nn.DataParallel(netG_128, device_ids=self.gpus)
            netG_256 = torch.nn.DataParallel(netG_256, device_ids=self.gpus)
            print(netG_64)
            state_dict = torch.load(cfg.TRAIN.NET_G_64,
                           map_location=lambda storage, loc: storage)  # 把所有的张量加载到CPU
            netG_64.load_state_dict(state_dict)
            state_dict_128 = torch.load(cfg.TRAIN.NET_G_128,
                           map_location=lambda storage, loc: storage)  # 把所有的张量加载到CPU
            netG_128.load_state_dict(state_dict_128)
            state_dict_256 = torch.load(cfg.TRAIN.NET_G_256,
                           map_location=lambda storage, loc: storage)  # 把所有的张量加载到CPU
            netG_256.load_state_dict(state_dict_256)
            print('Load ', cfg.TRAIN.NET_G_64)
            s_tmp = cfg.TRAIN.NET_G_256    # the path to save generated images
            istart = s_tmp.rfind('_') + 1
            iend = s_tmp.rfind('.')
            iteration = int(s_tmp[istart:iend])
            s_tmp = s_tmp[:s_tmp.rfind('/')]
            save_dir = '%s/iteration%d' % (s_tmp, iteration)
            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(self.batch_size, nz))
            if cfg.CUDA:
                netG_64.cuda()
                netG_128.cuda()
                netG_256.cuda()
                noise = noise.cuda()
            netG_64.eval()
            netG_128.eval()
            netG_256.eval()
            for step, data in enumerate(self.data_loader, 0):
                imgs, t_embeddings, filenames = data
                if cfg.CUDA:
                    t_embeddings = Variable(t_embeddings).cuda()
                else:
                    t_embeddings = Variable(t_embeddings)
                embedding_dim = t_embeddings.size(1)
                batch_size = imgs[0].size(0)
                noise.data.resize_(batch_size, nz)
                noise.data.normal_(0, 1)
                fake_img_list = []
                for i in range(embedding_dim):
                    fake_imgs_64, _, _,h_code_64 = netG_64(noise, t_embeddings[:, i, :])
                    fake_imgs_128, _, _, h_code_128 = netG_128(noise, h_code_64, t_embeddings[:, i, :])
                    fake_imgs_256, _, _, h_code_256 = netG_256(noise, h_code_128, t_embeddings[:, i, :])
                    if cfg.TEST.B_EXAMPLE:
                        # fake_img_list.append(fake_imgs[0].data.cpu())
                        # fake_img_list.append(fake_imgs[1].data.cpu())
                        fake_img_list.append(fake_imgs_256[0].data.cpu())
                    else:
                        self.save_singleimages(fake_imgs_256[0], filenames,
                                               save_dir, split_dir, i, 256)
                        # self.save_singleimages(fake_imgs[-2], filenames,
                        #                        save_dir, split_dir, i, 128)
                        # self.save_singleimages(fake_imgs[-3], filenames,
                        #                        save_dir, split_dir, i, 64)
                if cfg.TEST.B_EXAMPLE:
                    self.save_superimages(fake_img_list, filenames,
                                          save_dir, split_dir, 256)
