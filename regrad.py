# -*- coding: utf-8 -*-


#载入第三方库
import json
import torch
import glob
import matplotlib.pyplot as plt
import os
import numpy as np
import random
from torch.utils import data
#载入自定义的库
from image_pro import Image,DepthImage
from grasp_pro import Grasps


class Regrad(torch.utils.data.Dataset):
    #载入Regead数据集的类
    def __init__(self,file_dir,include_depth = True,include_rgb = True,start = 0.0,end = 1.0,ds_rotate = 0,random_rotate = False,random_zoom = False,output_size = 300):
        '''
        参数
        ----------
        file_dir : str
            存放Regead数据集的路径.
        include_depth : bool, optional
            描述. 是否包含深度图像数据，The default is True.
        include_rgb : bool, optional
            描述. 是否包含RGB图像数据The default is True.
        start : float, optional
            描述. 数据集划分起点参数，The default is 0.0.
        end : float, optional
            描述. 数据集划分终点参数，The default is 1.0.
        Returns
        -------
        None.
        '''
        super(Regrad,self).__init__()
        
        #一些参数的传递
        self.include_depth = include_depth
        self.include_rgb = include_rgb
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.output_size = output_size
        #去指定路径载入数据集数据
        graspf = glob.glob(os.path.join(file_dir,'*','*.json'))
        graspf.sort()
        l = len(graspf)
        if ds_rotate:
            graspf = graspf[int(l*ds_rotate):] + graspf[:int(l*ds_rotate)]

        if l == 0:
            raise FileNotFoundError('没有查找到数据集，请检查路径{}'.format(file_dir))
        if ds_rotate:
            graspf = graspf[int(l*ds_rotate):] + graspf[:int(l*ds_rotate)]
        rgbf = [filename.replace('.json','.jpg') for filename in graspf]
        depthf = [filename.replace('.json','.png') for filename in graspf]
        indexf = [filename.replace('.json', 'index.jpg') for filename in graspf]


        #按照设定的边界参数对数据进行划分并指定为类的属性
        self.graspf = graspf[int(l*start):int(l*end)]
        self.rgbf = rgbf[int(l*start):int(l*end)]
        self.depthf = depthf[int(l*start):int(l*end)]
        self.indexf = indexf[int(l*start):int(l*end)]

    @staticmethod
    def numpy_to_torch(s):
        '''
        :功能     :将输入的numpy数组转化为torch张量，并指定数据类型，如果数据没有channel维度，就给它加上这个维度
        :参数 s   :numpy ndarray,要转换的数组
        :返回     :tensor,转换后的torch张量
        '''
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def _get_crop_attrs(self, idx):
        '''
        :功能     :读取多抓取框中心点的坐标，并结合output_size计算要裁剪的左上角点坐标
        :参数 idx :int,
        :返回     :计算出来的多抓取框中心点坐标和裁剪区域左上角点坐标
        '''
        grasp_rectangles = Grasps.load_from_regrad_files(self.graspf[idx],scale_h = self.output_size/480.0,scale_w = self.output_size/640.0)
        center = grasp_rectangles.center
        #print (center)
        # 按照ggcnn里面的话，这里本该加个限制条件，防止角点坐标溢出边界，但前面分析过，加不加区别不大，就不加了
        # 分析错误，后面出现bug了，所以还是加上吧
        left = max(0, min(center[0] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[1] - self.output_size // 2, 480 - self.output_size))

        return center, left, top  # the center (column,row) must be changed to for later rotate considerations
        # the center I returned here is equal ti center[::-1] in ggcnn ,but the left and top params is totally equal to those in ggcnn,so all the
        # operation which use left and top could be write as same as ggcnn's

    def get_index(self,idx,rot = 0,zoom = 1.0,normalize = True):
        '''
        :功能     :读取返回指定id的index图像
        :参数 idx :int,要读取的数据id
        :返回     :ndarray,处理好后的index图像
        '''
        index_img = Image.from_file(self.indexf[idx])
        index_img.resize((480,640))
        center, left, top = self._get_crop_attrs(idx)
        index_img.rotate(rot,center)
        index_img.crop((top,left),(min(480,top+self.output_size),min(640,left+self.output_size)))
        index_img.zoom(zoom)
        index_img.resize((self.output_size, self.output_size))
        #index_img.img=index_img.img.astype('float32') / 255.0
        if normalize:
            index_img.normalize()
            #rgb_img.img = rgb_img.img.transpose((2, 0, 1))
            # 这里还有一句transpose没写，先不管
        return index_img.img

    def get_rgb_orign(self,idx,rot = 0,zoom = 1.0):
        '''
        :功能     :读取返回指定id的rgb图像
        :参数 idx :int,要读取的数据id
        :返回     :ndarray,处理好后的rgb图像
        '''
        rgb_img = Image.from_file(self.rgbf[idx])
        #第一次reasize,将图片从1280*960缩放到640*480,抓取框处理的时候也要处理两次
        rgb_img.resize((480, 640))

        #print(rgb_img.img.shape)
        #rgb_img.rotate(rot)
        center,left,top = self._get_crop_attrs(idx)
        #先旋转后裁剪再缩放最后resize
        rgb_img.rotate(rot,center)
        rgb_img.crop((top,left),(min(480,top+self.output_size),min(640,left+self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))

        return rgb_img.img
    
    def get_rgb(self,idx,rot = 0,zoom = 1.0,normalize = True):
        '''
        :功能     :读取返回指定id的rgb图像
        :参数 idx :int,要读取的数据id
        :返回     :ndarray,处理好后的rgb图像
        '''
        rgb_img = Image.from_file(self.rgbf[idx])
        #第一次reasize,将图片从1280*960缩放到640*480,抓取框处理的时候也要处理两次
        rgb_img.resize((480, 640))

        #print(rgb_img.img.shape)
        #rgb_img.rotate(rot)
        center,left,top = self._get_crop_attrs(idx)
        #先旋转后裁剪再缩放最后resize
        rgb_img.rotate(rot,center)
        rgb_img.crop((top,left),(min(480,top+self.output_size),min(640,left+self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))


        if normalize:
            rgb_img.normalize()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
            #这里还有一句transpose没写，先不管
        return rgb_img.img
        
    def get_depth(self,idx,rot = 0,zoom = 1.0):
        '''
        :功能     :读取返回指定id的depth图像
        :参数 idx :int,要读取的数据id
        :返回     :ndarray,处理好后的depth图像
        '''
        depth_img = DepthImage.from_tiff(self.depthf[idx])
        #print(depth_img.img.shape)
        #这里regrad数据集给出的png格式的深度图有三个通道,先给处理成二维的
        #depth_img.img = depth_img.img[0]
        #同理,先处理到640*480
        depth_img.resize((480, 640))
        #print(depth_img.img.shape)
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalize()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))

        return depth_img.img


        
    def get_grasp(self,idx,rot = 0,zoom=1.0):
        grs = Grasps.load_from_regrad_files(self.graspf[idx],scale_h = self.output_size/480.0,scale_w = self.output_size/640.0)#因为图像每个都resize了，所以这里每个抓取框都要缩放
        center, left, top = self._get_crop_attrs(idx)
        # 先旋转再偏移再缩放
        grs.rotate(rot, center)
        grs.offset((-left, -top))
        grs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        pos_img, angle_img, width_img = grs.generate_img(shape=(self.output_size, self.output_size))

        return pos_img,angle_img,width_img
        
    def get_raw_grasps(self,idx,rot = 0,zoom = 1.0):
        '''
        :功能       :读取返回指定id的抓取框信息斌进行一系列预处理(裁剪，缩放等)后以Grasps对象的形式返回
        :参数 idx   :int,要读取的数据id
        :返回       :Grasps，此id中包含的抓取
        '''
        raw_grasps = Grasps.load_from_regrad_files(self.graspf[idx],scale_h = self.output_size/480.0,scale_w = self.output_size/640.0)
        center, left, top = self._get_crop_attrs(idx)
        #print (left,top)
        raw_grasps.rotate(rot,center)
        #print (raw_grasps.points)
        raw_grasps.offset((-left,-top))
        #print (raw_grasps.points)
        raw_grasps.zoom(zoom,(self.output_size//2,self.output_size//2))
        return raw_grasps
    
    def __getitem__(self,idx):
        #一些参数的设置
        #随机旋转的角度在下面四个里面选，并不是完全随机
        if self.random_rotate:
            rotations = [0, np.pi/2, 2*np.pi/2, 3*np.pi/2]
            rot = random.choice(rotations)
        else:
            rot = 0.0
        #随机缩放的因子大小果然有限制，太大或者太小都会导致一些问题
        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.0
        # 载入深度图像
        #print(idx)
        if self.include_depth:
            depth_img = self.get_depth(idx,rot = rot,zoom = zoom_factor)
            x = self.numpy_to_torch(depth_img)
        # 载入rgb图像
        if self.include_rgb:
            rgb_img = self.get_rgb(idx,rot = rot,zoom = zoom_factor)
            #torch是要求channel-first的，检测一下，如果读进来的图片是channel-last就调整一下，ggcnn中目前我没看到在哪调整的，但肯定是做了的
            if rgb_img.shape[2] == 3:
                rgb_img = np.moveaxis(rgb_img,2,0)
            x = self.numpy_to_torch(rgb_img)
        if self.include_depth and self.include_rgb:#如果灰度信息和rgb信息都要的话，就把他们堆到一起构成一个四通道的输入，
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img,0),rgb_img),0
                )
            )

        #将原先的index标签进行处理，使其与深度图和ｒｇｂ图处理后的形式相似
        index_img = self.get_index(idx,rot = rot,zoom = zoom_factor)
        index_img = self.numpy_to_torch(index_img)
            
        # 载入抓取标注参数
        #pos_img,angle_img,width_img = self.get_grasp(idx,rot = rot,zoom = zoom_factor)

        # plt.imshow(pos_img)
        # plt.show()
        # 处理一下角度信息，因为这个角度值区间比较大，不怎么好处理，所以用两个三角函数把它映射一下：
        # cos_img = self.numpy_to_torch(np.cos(2*angle_img))
        # sin_img = self.numpy_to_torch(np.sin(2*angle_img))
        #
        # pos_img = self.numpy_to_torch(pos_img)
        
        # 限定抓取宽度范围并将其映射到[0,1]
        # width_img = np.clip(width_img, 0.0, 150.0)/150.0
        # width_img = self.numpy_to_torch(width_img)
        
        return x,index_img,idx,rot,zoom_factor#这里多返回idx,rot和zomm_facor参数，方便后面索引查找真实标注，并对真实的标注做同样处理以保证验证的准确性
    #映射类型的数据集，别忘了定义这个函数
    def __len__(self):
        return len(self.graspf)
