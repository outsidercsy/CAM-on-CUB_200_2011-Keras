# encoding: utf-8
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.layers import GlobalAveragePooling2D,Dense
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adagrad
import numpy as np
from keras.models import load_model  

import cv2
import  xml.dom.minidom
import os
import sys
import shutil

sys.setrecursionlimit(100000)


classname_id_link={}
for i in range(200):
    classname_id_link[str(i)] = i


def generate_bnd_box_from_segmentation(class_file_name,class_picture_name):
    segmentation=cv2.imread('./CUB_200_2011/CUB_200_2011/segmentations/'+class_file_name +'/'+ class_picture_name[:-4]+'.png' ,cv2.IMREAD_GRAYSCALE)
    row,col=segmentation.shape
    xmin=row-1 
    ymin=col-1 
    xmax=0
    ymax=0
    for i in range(row):
        for j in range(col):
            if segmentation[i][j]>0:
                if j<xmin:
                    xmin=j
                if j>xmax:
                    xmax=j
                if i<ymin:
                    ymin=i
                if i>ymax:
                    ymax=i
    return [xmin,ymin,xmax,ymax]

#归一化至0~1
def normalization_0_1(array):
    array_max=array.max()
    array_min=array.min()
    row,col=array.shape
    for i in range(row):
        for j in range(col):
            array[i][j]=(array[i][j]-array_min)/(array_max-array_min)
    return array

#从cam_mask中提取最大连通域
def generate_cam_main_mask(cam_mask):
    row,col=cam_mask.shape
    block_id=0
    cam_classified=np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            if i%10==0 and j%10==0:
                if cam_mask[i][j]==1 and cam_classified[i][j]==0:
                    block_id=block_id+1
                    expand(cam_mask,cam_classified,i,j,block_id,row,col)
    
    #对每个block的像素个数
    cnt_dict={}
    for i in range(row):
        for j in range(col):
            cnt_dict.setdefault(cam_classified[i][j],0)
            cnt_dict[cam_classified[i][j]]+=1
    cnt_dict.pop(0)

    main_block_id=max(cnt_dict, key=cnt_dict.get)

    #仅保留最大的block
    cam_main_mask=np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            if cam_classified[i][j]==main_block_id:
                cam_main_mask[i][j]=1

    return cam_main_mask

def expand(cam_mask,cam_classified,x,y,block_id,row,col):
    for i in range(-5,5):
        for j in range(-5,5):
            if x+i>=0 and x+i<row and y+j>=0 and y+j<col:
                cam_classified[x+i][y+j]=block_id

    if x-10>=0:
        if cam_mask[x-10][y]==1 and cam_classified[x-10][y]==0:
            expand(cam_mask,cam_classified,x-10,y,block_id,row,col)
    if x+10<row:
        if cam_mask[x+10][y]==1 and cam_classified[x+10][y]==0:
            expand(cam_mask,cam_classified,x+10,y,block_id,row,col)
    if y-10>=0:
        if cam_mask[x][y-10]==1 and cam_classified[x][y-10]==0:
            expand(cam_mask,cam_classified,x,y-10,block_id,row,col)                    
    if y+10<col:
        if cam_mask[x][y+10]==1 and cam_classified[x][y+10]==0:
            expand(cam_mask,cam_classified,x,y+10,block_id,row,col)

#从cam_main_mask得到预测的bounding_box 
def generate_bounding_box(cam_main_mask):
    row,col=cam_main_mask.shape
    ymax=0
    ymin=row-1
    xmin=col-1
    xmax=0
    for i in range(row):
        for j in range(col):
            if cam_main_mask[i][j]==1:
                if i<ymin:
                    ymin=i
                if i>ymax:
                    ymax=i
                if j>xmax:
                    xmax=j
                if j<xmin:
                    xmin=j
    return [xmin,ymin,xmax,ymax]

def calculate_IoU(xmin_pre,ymin_pre,xmax_pre,ymax_pre,xmin_true,ymin_true,xmax_true,ymax_true,class_picture_name):
    if xmin_true<=xmin_pre and xmax_true>=xmax_pre:
        dx=xmax_pre-xmin_pre
    elif xmin_true>=xmin_pre and xmax_true<=xmax_pre:
        dx=xmax_true-xmin_true
    elif xmin_true>=xmin_pre and xmax_true>=xmax_pre:
        dx=xmax_pre-xmin_true  
    elif xmin_true<=xmin_pre and xmax_true<=xmax_pre:
        dx=xmax_true-xmin_pre

    if ymin_true<=ymin_pre and ymax_true>=ymax_pre:
        dy=ymax_pre-ymin_pre
    elif ymin_true>=ymin_pre and ymax_true<=ymax_pre:
        dy=ymax_true-ymin_true
    elif ymin_true>=ymin_pre and ymax_true>=ymax_pre:
        dy=ymax_pre-ymin_true  
    elif ymin_true<=ymin_pre and ymax_true<=ymax_pre:
        dy=ymax_true-ymin_pre

    if dx>0 and dy>0:
        nume=dx*dy
    else:
        nume=0
       
    deno= (xmax_pre-xmin_pre)*(ymax_pre-ymin_pre) +  (xmax_true-xmin_true)*(ymax_true-ymin_true) - dx*dy  

    IoU=1.0*nume/deno

    print 'class_picture_name = ' + class_picture_name + '    IoU = '+str(IoU)
   
    return IoU

def drawing_box(ori_image,xmin_pre,ymin_pre,xmax_pre,ymax_pre,xmin_true,ymin_true,xmax_true,ymax_true):
    box_image=ori_image
    for y in [ymin_true,ymax_true]:
        for x in range(xmin_true,xmax_true):
            box_image[y][x]=[0,255,0]
    for x in [xmin_true,xmax_true]:
        for y in range(ymin_true,ymax_true):
            box_image[y][x]=[0,255,0]   

    for y in [ymin_pre,ymax_pre]:
        for x in range(xmin_pre,xmax_pre):
            box_image[y][x]=[0,0,255]
    for x in [xmin_pre,xmax_pre]:
        for y in range(ymin_pre,ymax_pre):
            box_image[y][x]=[0,0,255]    
    return box_image

def one_predict(model,w,classname_id_link,class_file_name,class_picture_name,cla_id,bnd_box_true):
    #一次预测，fmap的shape为(14, 14, 1024)
    ori_image=cv2.imread('./CUB_200_2011/CUB_200_2011/validation/'+class_file_name +'/'+ class_picture_name ,cv2.IMREAD_COLOR)
    row,col,height=ori_image.shape
    #resize输入的图像到（224，224，3）
    I1=cv2.resize(ori_image,(224,224),interpolation=cv2.INTER_CUBIC)

    I1 = I1[np.newaxis, :]  
    fmap,pro=model.predict(x=I1,batch_size=1)
    fmap=fmap[0]

    xmin_true=bnd_box_true[0]   
    ymin_true=bnd_box_true[1]  
    xmax_true=bnd_box_true[2]
    ymax_true=bnd_box_true[3]

    #生成class activation map    
    fmap=fmap.transpose((2,0,1))
    w=w.transpose((1,0))
    cam_cla=np.zeros((14,14))
    for i in range(1024):
        cam_cla+=fmap[i]*w[cla_id][i]

    #放大cam至原图尺寸
    cam_cla=cv2.resize(cam_cla,(col,row))#,interpolation=cv2.INTER_CUBIC
    
    #归一化
    cam_cla=normalization_0_1(cam_cla)

    #通过排序得到60%分位值
    aline=cam_cla.reshape((row*col))
    aline=aline.tolist()
    aline=sorted(aline)

    #二值化   阈值为   60%分位值 和  0.2   之中较大的一个
    if aline[int(0.6*row*col)] > 0.2:
        threshold=aline[int(0.6*row*col)]
    else:
        threshold=0.2
    print 'threshold='+str(threshold)
    cam_mask=np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            if cam_cla[i][j]>threshold:
                cam_mask[i][j]=1

    #生成bounding box
    cam_main_mask=generate_cam_main_mask(cam_mask)
    xmin_pre,ymin_pre,xmax_pre,ymax_pre=generate_bounding_box(cam_main_mask)

    box_image=drawing_box(ori_image,xmin_pre,ymin_pre,xmax_pre,ymax_pre,xmin_true,ymin_true,xmax_true,ymax_true)

    #计算mIoU
    IoU=calculate_IoU(xmin_pre,ymin_pre,xmax_pre,ymax_pre,xmin_true,ymin_true,xmax_true,ymax_true,class_picture_name)
      
    #保存效果图
    cv2.imwrite('./bounding_box_generate/'+ class_picture_name[:-4] + '-hotmap.jpg', 255*cam_cla)
    cv2.imwrite('./bounding_box_generate/'+ class_picture_name[:-4] + '-mask.jpg',255*cam_mask)
    cv2.imwrite('./bounding_box_generate/'+ class_picture_name[:-4] + '-main_mask.jpg',255*cam_main_mask)
    cv2.imwrite('./bounding_box_generate/'+ class_picture_name[:-4] + '-bnd_box.jpg', box_image)
    return IoU



#载入model
model = load_model('./cam_model.h5')
#载入全连接层参数，w的shape为(1024，20)
w=np.load("./fcn_w.npy")



mIoU=0
over_half_num=0

#200个类别，每个类别取一张图计算IoU
class_file_names = os.listdir('./CUB_200_2011/CUB_200_2011/validation')
for class_file_name in class_file_names:
    if class_file_name!='.DS_Store':
        cla_id=int(class_file_name[0:3])-1
        class_file_name_seperate=class_file_name.split('.')
        class_picture_names = os.listdir('./CUB_200_2011/CUB_200_2011/validation/'+class_file_name)
        for class_picture_name in class_picture_names[4:5]:
            print class_picture_name
            if class_picture_name!='.DS_Store':
                bnd_box_true=generate_bnd_box_from_segmentation(class_file_name,class_picture_name)
                IoU=one_predict(model,w,classname_id_link,class_file_name,class_picture_name,cla_id,bnd_box_true)
                if IoU>0.5:
                    over_half_num+=1
                mIoU+=IoU


mIoU/=200
over_half_rate=over_half_num/200.
print '0.5IoU accuracy=' + str(over_half_rate)
print 'mIoU=' +str(mIoU)

cv2.waitKey(0) #等待按键


