import numpy as np
import math
import os
import pickle
import cv2
import skimage
from math import floor
from PIL import Image
from funcs import hashtable,CT

def psnr(target, ref):
    # target:目标图像  ref:参考图像  
    h,w=ref.shape
    diff=np.zeros((h,w))+ref-target
    rmse=math.sqrt(np.mean(diff**2.))
    
    return 20*math.log10(255.0/rmse)


result=open('./evaluate/result.txt','a')

with open('filter160','rb') as fp:
    h=pickle.load(fp)

trainimgs=[]
for root,dirs,files in os.walk("./evaluate"):
    for file in files:
        if file.lower().endswith(('jpg','bmp','png')):
            trainimgs.append(os.path.join(root,file))
#open images in folder

for imgpath in trainimgs:
 
    print('\nevaluate:',imgpath[11:-4])
    img=Image.open(imgpath).convert('YCbCr')#YCbCr mode to record Y


    originY=np.array(img)[:,:,0]#ndarray to save the luminance channel
    n_originY=cv2.normalize(originY.astype('float'),None,originY.min()/255,originY.max()/255,cv2.NORM_MINMAX)
    #the orignial lightness information
    
    LRimg=img.resize((img.size[0]//2,img.size[1]//2))#img type

    HR=np.zeros((LRimg.size[1]*2-10,LRimg.size[0]*2-10))
    FinalPic=np.zeros((LRimg.size[1]*2-12,LRimg.size[0]*2-12))


    initLRimg=LRimg.resize((LRimg.size[0]*2,LRimg.size[1]*2),Image.BILINEAR)#img type
    initLRimg.save('./evaluate_save/'+imgpath[11:-4]+'_initial.jpg')

    LRarray=np.array(initLRimg)#all pixels
    LR=np.array(LRarray[5:LRimg.size[1]*2-5,5:LRimg.size[0]*2-5,0])#deepcopy
        
    init=LRarray[:,:,0]#ndarray to save the bilinear pic luminance channel

    n_init=cv2.normalize(init.astype('float'),None,init.min()/255,init.max()/255,cv2.NORM_MINMAX)
    #resize the half then twice the size with bilinear interpolation
    ht,wd=n_init.shape
    
    for row in range(5,ht-5):
        if row%20==0 or row==ht-6:
            process=int((row/(ht-6))*100)
            print('\r%{:3}['.format(process)+'*'*(process//5)+'->'+'.'*(20-process//5)+']',end='')
        for column in range(5,wd-5):
            patch=n_init[row-5:row+6,column-5:column+6]
            angle,strength,coherence=hashtable(patch)
            patch=np.matrix(patch.ravel())
         
            filtertype=((row-5)%2)*2+(column-5)%2

            HR[row-5,column-5]=patch.dot(h[angle,strength,coherence,filtertype])

    HR=np.clip(HR.astype('float')*255.0,0.0,255.0)#recover and limit the value
    
    FinalPic=CT(HR,LR)
    ht,wt=originY.shape
    if ht%2==1:
        ht-=1
    if wt%2==1:
        wt-=1
    print('\nevaluate result for :',imgpath[11:-4])
    print('PSNR of raisr:',psnr(FinalPic,originY[6:ht-6,6:wt-6]))
    print('PSNR of bilinear:',psnr(init[6:-6,6:-6],originY[6:ht-6,6:wt-6]))
    print('SSIM of raisr:',skimage.measure.compare_ssim(FinalPic,originY[6:ht-6,6:wt-6]))
    print('SSIM of bilinear:',skimage.measure.compare_ssim(init[6:-6,6:-6],originY[6:ht-6,6:wt-6]))
    
    print('evaluate result for :',imgpath[11:-4],file=result)
    print('PSNR of raisr:',psnr(FinalPic,originY[6:ht-6,6:wt-6]),file=result)
    print('PSNR of bilinear:',psnr(init[6:-6,6:-6],originY[6:ht-6,6:wt-6]),file=result)
    print('SSIM of raisr:',skimage.measure.compare_ssim(FinalPic,originY[6:ht-6,6:wt-6]),file=result)
    print('SSIM of bilinear:',skimage.measure.compare_ssim(init[6:-6,6:-6],originY[6:ht-6,6:wt-6]),file=result,end='\n\n')
    try:
        LRarray[6:LRimg.size[1]*2-6,6:LRimg.size[0]*2-6,0]=FinalPic
    except:
        pass
    new=Image.fromarray(LRarray,'YCbCr')

    new.save('./evaluate_save/'+imgpath[11:-4]+'_upscaled.jpg')

print('%'*25,file=result)
result.close()

         
    
