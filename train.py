import cv2
import os
import numpy as np
import pickle

from PIL import Image
from funcs import *

trainimgs=[]
for root,dirs,files in os.walk("./train"):
    for file in files:
        if file.lower().endswith(('jpg','bmp','png','jpeg')):
            trainimgs.append(os.path.join(root,file))
#open images in folder

Q=np.zeros((24,3,3,2*2,11*11,11*11))
#24 : Quantize angle into 24 buckets
#3 : Quantize strength
#3 : Quantize coherence
#2*2 : upscale by 2:needs 4 filters for different locations of bilinear interpolation 
#11*11 : patch size
V=np.zeros((24,3,3,2*2,11*11))
h=np.zeros((24,3,3,2*2,11*11))#filter to learn
for imgpath in trainimgs:
    print('\ntraining:',imgpath)
    img=Image.open(imgpath).convert('YCbCr')#YCbCr mode to record Y
    originY=np.array(img)[:,:,0]#ndarray to save the luminance channel
    originY=cv2.normalize(originY.astype('float'),None,originY.min()/255,originY.max()/255,cv2.NORM_MINMAX)
    #the orignial lightness information
    LR=img.resize((img.size[0]//2,img.size[1]//2),Image.BICUBIC)#img type
    initLR=LR.resize((LR.size[0]*2,LR.size[1]*2),Image.BILINEAR)#img type
    init=np.array(initLR)[:,:,0]#ndarray
    init=cv2.normalize(init.astype('float'),None,init.min()/255,init.max()/255,cv2.NORM_MINMAX)
    #resize the half then twice the size with bilinear interpolation
    ht,wd=init.shape
    for row in range(5,ht-5):
        if row%20==0 or row==ht-6:
            process=int((row/(ht-6))*100)
            print('\r%{:3}['.format(process)+'*'*(process//5)+'->'+'.'*(20-process//5)+']',end='')
        for column in range(5,wd-5):
            patch=init[row-5:row+6,column-5:column+6]
            angle,strength,coherence=hashtable(patch)
            patch=np.matrix(patch.ravel()) #row vector corresponding to A
            #hashblock=init[row-4:row+5,column-4:column+5] #hashblock 9*9 in paper         
            filtertype=((row-5)%2)*2+(column-5)%2
            b=originY[row][column]
            Q[angle,strength,coherence,filtertype]+=np.dot(patch.T,patch)
            #Q=A'.A   #V=A'.b
            tmp=np.dot(patch.T,b)
            tmp=np.array(tmp).ravel()
            V[angle,strength,coherence,filtertype]+=tmp

print('\ncalculating filter h...')
for ftype in range(4):
    for an in range(24):
        process=int((((an+1)+24*ftype)/(24*4))*100)
        print('\r%{:3}['.format(process)+'*'*(process//5)+'->'+'.'*(20-process//5)+']',end='')
        for st in range(3):
            for co in range(3):
                h[an,st,co,ftype]=cg(Q[an,st,co,ftype],V[an,st,co,ftype])


with open('filter160','wb') as fp:
    pickle.dump(h,fp)





            
            
    
