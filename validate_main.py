# -*- coding: utf-8 -*-
"""


验证函数一次预测是否正确的函数，这里仅仅是示例，一个测试而已，还没有将它集成到一个完整的训练过程中去，集成到训练过程中去的程序见第二部分

这个程序运行可能会报错：'No module named 'train',这是pytorch的一个bug，是因为载入的训练好的模型的路径发生变化导致的，目前也没有什么好办法解决，就直接训练一个新的，重新放到这个文件夹就行


"""
import numpy as np
import torch
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
#自定义包
from torch.utils import data
from regrad import Regrad
from functions import post_process,detect_grasps,max_iou
from grasp_pro import Grasps
#要验证网络性能，就必须要有一个训练完的网络，这里不从头训练，直接载入一个训练好的模
net = torch.load('/home/user/xyh_rl_projects/ggcnn_xyh/10/index_model1')

#设置训练设备
device = torch.device('cuda:0')

#构建数据集并获取一组数据
dataset = Regrad('/home/user/xyh_rl_projects/new_data')
val_data = torch.utils.data.DataLoader(dataset,shuffle = True,batch_size = 1)
regrad_path = "/home/user/xyh_rl_projects/new_data"
graspf = glob.glob(os.path.join(regrad_path,'*','*.json'))
#读出一组数据及其编号

for x, y, id_x, _, _ in val_data:
    xc = x.to(device)
    yc = y.to(device)
    idx = id_x
    print(idx)
    break

rgb_img = val_data.dataset.get_rgb_orign(idx)
#rgb_img = rgb_img.cpu().data.numpy().squeeze()*255
depth_img = val_data.dataset.get_depth(idx)
index_img_true = val_data.dataset.get_index(idx)
plt.subplot(221)
plt.imshow(rgb_img)
plt.subplot(222)
plt.imshow(depth_img)
plt.subplot(223)

plt.imshow(index_img_true)



#输入网络计算预测结果
net = net.to(device)
#pos_img,cos_img,sin_img,width_img = net(xc)
index_img = net(xc)
#原始输出的预处理
index_img = index_img.cpu().data.numpy().squeeze()*255

plt.subplot(224)
plt.imshow(index_img)


plt.show()
