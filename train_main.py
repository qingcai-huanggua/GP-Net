#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 15:32:02 2020

#这个程序是在一个batch上测试的程序，只在一个样本上训练的，并没有遍历整个数据集，train_main2.py是遍历了整个数据集的

"""

# 导入第三方包
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import tensorboardX

import time
import datetime
import os
# summary输出保存之后print函数失灵，但还可以用这个logging来打印输出
import logging
import sys
# 导入自定义包
from regrad import Regrad
from ggcnn2 import GGCNN2 as GPNet
#from cornell_pro import Cornell
from functions import post_process, detect_grasps, max_iou



batch_size = 96

# 准备数据集
cornell_data = Regrad('/home/user/xyh_rl_projects/new_data')
dataset = torch.utils.data.DataLoader(cornell_data, batch_size=batch_size)

# 从数据集中读取一个样本
for x, y, _, _, _ in dataset:
    xc = x
    yc = y
    break

l = 0
# 简单可视化输入一下
depth_img = xc[l][0].data.numpy()
rgb_img = xc[l][0:3].data.numpy()

rgb_img = np.moveaxis(rgb_img, 0, 2) * 255

index_img = yc[l][0].data.numpy() * 255
plt.figure()

plt.subplot(131)
plt.title('depth_input')
plt.imshow(depth_img)
plt.subplot(132)
plt.title('rgb_input')
plt.imshow(rgb_img)
plt.subplot(133)
plt.imshow(index_img)

plt.savefig("/home/user/xyh_rl_projects/picture/index_input(%d).jpg"%l)
plt.show()



j = 78
# 简单可视化输入一下
depth_img = xc[j][0].data.numpy()
rgb_img = xc[j][0:3].data.numpy()

rgb_img = np.moveaxis(rgb_img, 0, 2) * 255

index_img = yc[j][0].data.numpy() * 255
plt.figure()

plt.subplot(131)
plt.title('depth_input')
plt.imshow(depth_img)
plt.subplot(132)
plt.title('rgb_input')
plt.imshow(rgb_img)
plt.subplot(133)
plt.imshow(index_img)

plt.savefig("/home/user/xyh_rl_projects/picture/index_input(%d).jpg"%j)
plt.show()

k = 87
# 简单可视化输入一下
depth_img = xc[k][0].data.numpy()
rgb_img = xc[k][0:3].data.numpy()

rgb_img = np.moveaxis(rgb_img, 0, 2) * 255

index_img = yc[k][0].data.numpy() * 255
plt.figure()

plt.subplot(131)
plt.title('depth_input')
plt.imshow(depth_img)
plt.subplot(132)
plt.title('rgb_input')
plt.imshow(rgb_img)
plt.subplot(133)
plt.imshow(index_img)

plt.savefig("/home/user/xyh_rl_projects/picture/index_input(%d).jpg"%k)
plt.show()
# print(img)
# grs = get_rectangles(grasp_path)
# plt.figure(figsize=(40, 20))
# plt.subplot(121)
# plt.imshow(rgb_img)

# for gr in grs[::]:
#     color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#     for i in range(3):
#         cv2.line(img, tuple(gr.points.astype(np.uint32)[i]), tuple(gr.points.astype(np.uint32)[i + 1]), color, 2)
#     img = cv2.line(img, tuple(gr.points.astype(np.uint32)[3]), tuple(gr.points.astype(np.uint32)[0]), color, 2)
# plt.subplot(122)
# plt.imshow(img)
# plt.show()






# 实例化一个网络
net = GPNet(4)

# 定义一个优化器
optimizer = optim.Adam(net.parameters())

# 设置GPU设备
device = torch.device("cuda:1")

net = net.to(device)

x = xc.to(device)
#y = [yy.to(device) for yy in yc]
y = yc.to(device)
#print (y.shape)
#print(x.shape)
# 动态显示每次优化过后的预测结果
fig = plt.figure()
plt.ion()
plt.show()

# 想要查看的结果编号num<batch_size


loss_results = []
# max_results = []
# width_results = []
index_results=[]
for i in range(1200):
    losses = net.compute_loss(x, y)
    loss = losses['loss']
    # 反向传播优化
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    #pos, cos, sin, width = net.forward(x)
    index = net.forward(x)
    loss_results.append(loss)
    #max_results.append(pos.cpu().data.numpy().max())
    index_results.append(index.cpu().data.numpy().max())
    if i % 2 == 0:
        print(loss)
        plt.cla()
        # pos = pos.cpu()
        # cos = cos.cpu()
        # sin = sin.cpu()
        # width = width.cpu()
        index = index.cpu()
        plt.subplot(111)
        plt.title('index_out')
        plt.imshow(index[l][0].data.numpy()*255, cmap=plt.cm.gray)
        plt.pause(0.01)

        # 拿它做个预测试试
        fig.suptitle('epoch:{0}\n loss:{1} \n max:{2}'.format(i, loss, index.cpu().data.numpy().max()))
        plt.savefig("/home/user/xyh_rl_projects/picture/index_output(%d).jpg"%l)
        plt.subplot(111)
        plt.title('index_out')
        plt.imshow(index[k][0].data.numpy()*255, cmap=plt.cm.gray)
        plt.pause(0.01)

        # 拿它做个预测试试
        fig.suptitle('epoch:{0}\n loss:{1} \n max:{2}'.format(i, loss, index.cpu().data.numpy().max()))
        plt.savefig("/home/user/xyh_rl_projects/picture/index_output(%d).jpg"%k)
        plt.subplot(111)
        plt.title('index_out')
        plt.imshow(index[j][0].data.numpy()*255, cmap=plt.cm.gray)
        plt.pause(0.01)

        # 拿它做个预测试试
        fig.suptitle('epoch:{0}\n loss:{1} \n max:{2}'.format(i, loss, index.cpu().data.numpy().max()))
        plt.savefig("/home/user/xyh_rl_projects/picture/index_output(%d).jpg"%j)
        # plt.subplot(141)
        # plt.title('pos_out')
        # plt.imshow(pos[num][0].data.numpy(), cmap=plt.cm.gray)
        # plt.subplot(142)
        # plt.title('cos_out')
        # plt.imshow(cos[num][0].data.numpy(), cmap=plt.cm.gray)
        # plt.subplot(143)
        # plt.title('sin_out')
        # plt.imshow(sin[num][0].data.numpy(), cmap=plt.cm.gray)
        # plt.subplot(144)
        # plt.title('width_out')
        # plt.imshow(width[num][0].data.numpy(), cmap=plt.cm.gray)


fig2 = plt.figure()
fig2.suptitle('loss and q_img & width_img max value')
plt.plot(loss_results, label='loss')
plt.plot(index_results, label='index_max')
#plt.plot(width_results, label='width_img_max')

plt.legend()

plt.savefig("/home/user/xyh_rl_projects/picture/index_loss.jpg")
plt.show()
torch.save(net, 'index_model1')
