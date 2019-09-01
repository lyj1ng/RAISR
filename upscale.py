import os
import numpy as np
import pickle
import cv2 
from math import floor
from PIL import Image
from funcs import hashtable,CT



with open('filter160','rb') as fp:
    h=pickle.load(fp)

trainimgs=[]
for root,dirs,files in os.walk("./toupscale"):
    for file in files:
        if file.lower().endswith(('jpg','bmp','png')):
            trainimgs.append(os.path.join(root,file))
#open images in folder

for imgpath in trainimgs:
 
    print('\nupscaling:',imgpath[12:-4])
    img=Image.open(imgpath).convert('YCbCr')#YCbCr mode to record Y

    initLR=img.resize((img.size[0]*2,img.size[1]*2),Image.BILINEAR)#img type
    initLR.save('./save/'+imgpath[12:-4]+'_initial.jpg')
    HR=np.zeros((img.size[1]*2-10,img.size[0]*2-10))

    LRarray=np.array(initLR)#all pixels
    LR=np.array(LRarray[5:img.size[1]*2-5,5:img.size[0]*2-5,0])#deepcopy

    init=LRarray[:,:,0]
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
            patch=np.matrix(patch.ravel())
         
            filtertype=((row-5)%2)*2+(column-5)%2

            HR[row-5,column-5]=patch.dot(h[angle,strength,coherence,filtertype])

    HR=np.clip(HR.astype('float')*255.0,0.0,255.0)#recover and limit the value
    
    LRarray[6:img.size[1]*2-6,6:img.size[0]*2-6,0]=CT(HR,LR)

    new=Image.fromarray(LRarray,'YCbCr')

    new.save('./save/'+imgpath[12:-4]+'_upscaled.jpg')

         
    
