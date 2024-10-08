#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import cv2
import torch.optim as optim
from torch.utils import data
from regrad import Regrad
from ggcnn2 import GGCNN2 as GPNET
import matplotlib.pyplot as plt
import numpy as np
batch_size = 96
batches_per_epoch = 100
epochs = 100
lr = 0.0001


def train(epoch,net,device,train_data,optimizer,batches_per_epoch):
    """
    :功能: 执行一个训练epoch
    :参数: epoch             : 当前所在的epoch数
    :参数: net               : 要使用的网络模型
    :参数: device            : 训练使用的设备
    :参数: train_data        : 训练所用数据集
    :参数: optimizer         : 优化器
    :参数: batches_per_epoch : 每个epoch训练中所用的数据批数

    :返回: 本epoch的平均损失
    """
    #结果字典，最后返回用
    results = {
        'loss': 0,
        'losses': {
        }
    }
    
    #训练模式，所有的层和参数都会考虑进来，eval模式下比如dropout这种层会不使能
    net.train()
    
    batch_idx = 0
    
    #开始样本训练迭代
    while batch_idx < batches_per_epoch:
        for x, y,idx,_,_,in train_data:
            batch_idx += 1
            xc = x
            idx=idx
            yc = y
            if batch_idx >= batches_per_epoch:
                break

            #
            # pos_img = yc[0][i][0].data.numpy()
            # cos_img = yc[1][i][0].data.numpy()
            # sin_img = yc[2][i][0].data.numpy()
            # width_img = yc[3][i][0].data.numpy()
            #
            # plt.figure()
            #
            # plt.subplot(231)
            # plt.title('depth_input')
            # plt.imshow(depth_img)
            # plt.subplot(232)
            # plt.title('rgb_input')
            # plt.imshow(rgb_img)
            # plt.subplot(233)
            # plt.title('pos_input')
            # plt.imshow(pos_img)
            # plt.subplot(234)
            # plt.title('cos_input')
            # plt.imshow(cos_img)
            # plt.subplot(235)
            # plt.title('sin_input')
            # plt.imshow(sin_img)
            # plt.subplot(236)
            # plt.title('width_input')
            # plt.imshow(width_img)
            # plt.savefig("/home/user/xyh_rl_projects/ggcnn_xyh/input(i).jpg")
            # plt.show()


            #将数据传到GPU
            xc = x.to(device)
            yc = y.to(device)
            
            lossdict = net.compute_loss(xc,yc)
            
            #获取当前损失
            loss = lossdict['loss']
            
            #打印一下训练过程
            if batch_idx % 10 == 0:
                print('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))
            
            #记录总共的损失
            results['loss'] += loss.item()
            #单独记录各项损失，pos,cos,sin,width
            for ln, l in lossdict['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()
            
            #反向传播优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        #计算总共的平均损失
        results['loss'] /= batch_idx
        
        #计算各项的平均损失
        for l in results['losses']:
            results['losses'][l] /= batch_idx
        torch.save(net, '../10/trained_model1')
        return results
def run(net):
    #获取设备
    device = torch.device("cuda:1")
    
    #实例化一个网络
    #net = GGCNN(4)
    net = net.to(device)
    
    #准备数据集
    cornell_data =Regrad('/home/user/xyh_rl_projects/new_data')
    dataset = torch.utils.data.DataLoader(cornell_data,batch_size = batch_size,shuffle = True)
    
    #设置优化器
    optimizer = optim.Adam(net.parameters())
    
    #开始主循环
    for epoch in range(epochs):
        train_results = train(epoch, net, device, dataset, optimizer, batches_per_epoch)
    
    return train_results
if __name__ == '__main__':
    #这块先把网络在外部定义，方便导出，后面写了保存函数就可以直接放在里面了
    net = GPNET(4)
    run(net)
    torch.save(net,'../10/trained_model_ggcnn2')
