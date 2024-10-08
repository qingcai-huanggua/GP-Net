#!/usr/bin/python  
# -*- coding: UTF-8 -*-
import os
import glob
import cv2
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from PIL import Image  
from PIL import ImageDraw  
from PIL import ImageFont  
from grasp_pro import Grasp_cpaw
from image_pro import Image,DepthImage
from grasp_pro import Grasps
regrad_path = "/home/user/xyh_rl_projects/new_data"
graspf = glob.glob(os.path.join(regrad_path,'*','*.json'))

graspf.sort()
#print(graspf[0])
rgbf = [filename.replace('.json','.jpg') for filename in graspf]
#print(rgbf[0])
depthf = [filename.replace('.json','.png') for filename in graspf]
def str2num(point,offset=(0, 0)):
    '''
    :功能  :将字符串类型存储的抓取框脚点坐标取整并以元组形式返回
    
    :参数  :point,字符串，以字符串形式存储的一个点的坐标
    :返回值 :列表，包含int型抓取点数据的列表[x,y]
    '''
    x=point[0]
    y=point[1]
    x,y = int(round(float(x))),int(round(float(y)))
    #return [int(round(float(y))) - offset[0], int(round(float(x))) - offset[1]]
    
    return (x,y)#如果想在后面可视化框的话，这里就得返回元组类型的数据，或者后面再类型转换为元组

def get_rectangles(regrad_grasp_file):
    '''
    :功能  :从抓取文件中提取抓取框的坐标信息
    
    :参数  :cornell_grap_file:字符串，指向某个抓取文件的路径
    :返回值 :列表，包含各个抓取矩形数据的列表
    '''
    grasp_rectangles = []


    with open(regrad_grasp_file,'r') as f:
        data = json.loads(f.read())  # load的传入参数为字符串类型
        #print(data, type(data))
        for value in data:
            grasp=value[1]

            x, y, w, h , theta = grasp[0][0],grasp[0][1],grasp[1][0],grasp[1][1],grasp[2]
            grasp_rectangles.append(Grasp_cpaw(np.array([x,y]),-theta/180.0*np.pi,h,w).as_gr)#我这边读取的顺序跟GGCNN中的有些不同,now.they asr totally same.
            #print(label0,label1)
            #point0 = f.readline().strip()
            #if not point0:
            #    break
            #point1,point2,point3 = f.readline().strip(),f.readline().strip(),f.readline().strip()
  
    #print(label0s)
    return grasp_rectangles

def draw_rectangles(img_path,grasp_path,depth_path,idx):
    '''
    :功能  :在指定的图片上绘制添加相应的抓取标注框
    
    :参数  :img_path:字符串，指向某个RGB图片的路径
    :参数  :grasp_path:字符串，指向某个抓取文件的路径
    :返回值 :numpy数组，已经添加完抓取框的img数组

    '''

    plt.figure(figsize=(15,15))
    depth_img = imread(depth_path)
    #print(depth_img)
    plt.imshow(depth_img)
    plt.savefig("/home/user/xyh_rl_projects/picture/regrad_depth(%d).jpg"%idx)


    img = cv2.imread(img_path)
    #print(img)
    grs = get_rectangles(grasp_path)
    plt.figure(figsize = (40,20))
    plt.subplot(121)
    plt.imshow(img)

    for gr in grs[::]:
        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        for i in range(3):
            cv2.line(img,tuple(gr.points.astype(np.uint32)[i]),tuple(gr.points.astype(np.uint32)[i+1]),color,2)
        img = cv2.line(img,tuple(gr.points.astype(np.uint32)[3]),tuple(gr.points.astype(np.uint32)[0]),color,2) 
    plt.subplot(122)
    plt.imshow(img)



    
    return img


def show(idx,output_size = 300,rotate = 0,zoom = 0.8):
    
    
    grs = Grasps.load_from_regrad_files(graspf[idx],scale_h = 300.0/480.0,scale_w = 300.0/640.0)

    img = Image.from_file(rgbf[idx])
    img.resize((480, 640))
    plt.subplot(111)
    plt.imshow(img.img)
    plt.show()
    # print(rgb_img.img.shape)
    # rgb_img.rotate(rot)
    grasp_rectangles = Grasps.load_from_regrad_files(graspf[idx], scale_h=300/ 480.0,scale_w=300 / 640.0)
    center = grasp_rectangles.center
    print (center)
    # 按照ggcnn里面的话，这里本该加个限制条件，防止角点坐标溢出边界，但前面分析过，加不加区别不大，就不加了
    # 分析错误，后面出现bug了，所以还是加上吧
    left = max(0, min(center[0] - 300 // 2, 640 - 300))
    top = max(0, min(center[1] - 300 // 2, 480 - 300))
    print(left,top)
    #grasp_rectangles.rotate(rot, center)
    grs.offset((-left, -top))
    #grasp_rectangles.zoom(zoom, (self.output_size // 2, self.output_size // 2))
    # 先旋转后裁剪再缩放最后resize
    #img.rotate(rot, center)
    img.crop((top, left), (min(480, top + 300), min(640, left + 300)))
    #rgb_img.zoom(zoom)
    #img.resize((300, 300))
    #img.suofang(640, 480)
    plt.figure(figsize = (15,30))
    #plt.subplot(211)
    #plt.imshow(rgb_img.img)
    for gr in grs.grs[::]:
        for i in range(3):
            print('gr.points{}'.format(gr.points))
            cv2.line(img.img,tuple(gr.points.astype(np.uint32)[i]),tuple(gr.points.astype(np.uint32)[i+1]),5)
        img.img = cv2.line(img.img,tuple(gr.points.astype(np.uint32)[3]),tuple(gr.points.astype(np.uint32)[0]),5)
    plt.subplot(111)
    plt.imshow(img.img)
    plt.show()


    # plt.figure(figsize = (10,10))
    # plt.subplot(121)
    # plt.title(u'处理前')
    # plt.imshow(img.img)
    # output_size = output_size
    # #计算一些图像处理需要的参数
    # center = grs.grs[0].center

    #print(center)
    #left = max(0, min(center[0] - output_size // 2, 1280 - output_size))
    #top = max(0, min(center[1] - output_size // 2, 960 - output_size))
    #print(left,top)
    #图像处理

    #img.rotate(rotate,center)

    #img.zoom(0.8)
    #img.crop((left,top),(left+output_size,top+output_size))
    # img.rotate(rotate,center)
    # img.zoom(zoom)
    # img.resize((output_size,output_size))
    # #显示图像
    # plt.subplot(122)
    # plt.title(u'处理后')
    # plt.imshow(img.img)
    # plt.show()

    # # 先测试一下图像读取这里
    # plt.figure(figsize=(15, 15))
    # idx = 10
    # rot = 0
    # zoom = 0.65
    # plt.subplot(131)
    # img = dataset.get_rgb(idx, rot=rot, zoom=zoom, normalise=False)
    # plt.imshow(im)
    # plt.subplot(132)
    # img = dataset.get_rgb(idx, rot=rot, zoom=zoom)
    # plt.imshow(img)
    # plt.subplot(133)
    # img = dataset.get_depth(idx, rot=rot, zoom=zoom)
    # plt.imshow(img)

    plt.show()
if __name__ == "__main__":

    #for i in range(0,40):
        #img = draw_rectangles(rgbf[i],graspf[i],depthf[i],i)
        #plt.savefig("/home/user/xyh_rl_projects/picture/regrad(%d).jpg"%i)

        #plt.show()
        #img = test(img)
#jupyter好像用不了 parser，所以这里就没用
    
    show(idx = 0)
    
#vmrd的数据集在后面训练时发现有问题，检查发现图片的规格并不是统一的，我在这里做一下旋转操作，将方向不对的图片调整过来，注意调整的时候实际坐标也要调整．代码集成在2里面．