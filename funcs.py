import numpy as np

from math import *

def hashtable(block):
    W=[0.0001,0.0002,0.0006,0.0011,0.0016,0.0018,0.0016,0.0011,0.0006,0.0002,0.0001
    ,0.0002,0.0007,0.0018,0.0033,0.0048,0.0054,0.0048,0.0033,0.0018,0.0007,0.0002
    ,0.0006,0.0018,0.0042,0.0079,0.0115,0.0131,0.0115,0.0079,0.0042,0.0018,0.0006
    ,0.0011,0.0033,0.0079,0.0148,0.0215,0.0244,0.0215,0.0148,0.0079,0.0033,0.0011
    ,0.0016,0.0048,0.0115,0.0215,0.0313,0.0355,0.0313,0.0215,0.0115,0.0048,0.0016
    ,0.0018,0.0054,0.0131,0.0244,0.0355,0.0402,0.0355,0.0244,0.0131,0.0054,0.0018
    ,0.0016,0.0048,0.0115,0.0215,0.0313,0.0355,0.0313,0.0215,0.0115,0.0048,0.0016
    ,0.0011,0.0033,0.0079,0.0148,0.0215,0.0244,0.0215,0.0148,0.0079,0.0033,0.0011
    ,0.0006,0.0018,0.0042,0.0079,0.0115,0.0131,0.0115,0.0079,0.0042,0.0018,0.0006
    ,0.0002,0.0007,0.0018,0.0033,0.0048,0.0054,0.0048,0.0033,0.0018,0.0007,0.0002
    ,0.0001,0.0002,0.0006,0.0011,0.0016,0.0018,0.0016,0.0011,0.0006,0.0002,0.0001]
    #generate by matlab(fspecial:gaussian low pass filter 11X11)
    W=np.diag(W)
    gy,gx=np.gradient(block)
    G=np.matrix((gx.ravel(),gy.ravel())).T
    x=G.T.dot(W).dot(G)
    eigenvalues,eigenvectors=np.linalg.eig(x)
    i=eigenvalues.argmax()
    ###eigenvector φk 1, corresponding to the largest eigenvalue
    
    theta_k=np.math.atan2(eigenvectors[1,i],eigenvectors[0,i])
    
    if theta_k<0:
        theta_k+=np.pi
    lamda_k_1=eigenvalues.max()
    k_1=np.math.sqrt(lamda_k_1)
    lamda_k_2=eigenvalues.min()
    mu_k=(k_1-np.math.sqrt(lamda_k_2))/(k_1+np.math.sqrt(lamda_k_2)+0.0001)

    angle=floor((theta_k/3.1416)*24)
    #if k_1>0.2:print(angle,'k',k_1)

    strength=floor((k_1/0.2)*3)#0.2-0
    coherence=floor(mu_k*3)#<=1
    if strength>2:
        strength=2
    elif strength<0:
        strength=0
        
    if coherence>2:
        coherence=2
    
    return angle,strength,coherence

def cg(Q,V):
    rslt=np.zeros((Q.shape[0]))    
    if Q.sum()<10 or np.linalg.det(Q)<=0:
        pass
    else:
        rslt=np.linalg.inv(Q).dot(V)
    return rslt


def CT(after,before):
    h,w=after.shape
    blend=np.zeros((h-2, w-2),dtype='uint8')
    censusA=np.zeros((h-2, w-2),dtype='uint8')
    censusB=np.zeros((h-2, w-2),dtype='uint8')

    #pixels to blend
    cpa=after[1:h-1,1:w-1]
    cpb=before[1:h-1,1:w-1]

    #offsets generation
    offsets=[(u, v) for v in range(3) for u in range(3) if not u == 1 == v]

    #Do the pixel comparisons
    for u,v in offsets:
        censusA = (censusA<<1) | (after[v:v+h-2, u:u+w-2] >= cpa)
        censusB = (censusB<<1) | (after[v:v+h-2, u:u+w-2] >= cpb)
    census=censusA ^ censusB
    print('\nCT calculating...')
    for r in range(census.shape[0]):
        if r%20==0 or r==census.shape[0]-1:
            process=int((r/(census.shape[0]-1))*100)
            print('\r%{:3}['.format(process)+'**'*(process//10)+'->'+'..'*(10-process//10)+']',end='')
        for c in range(census.shape[1]):
            count=0
            while census[r,c]!=0:
                if census[r,c] & 1:
                    count+=1
                census[r,c]>>=1
            census[r,c]=count

    print('\nCT blending...')
    for r in range(census.shape[0]):
        if r%20==0 or r==census.shape[0]-1:
            process=int((r/(census.shape[0]-1))*100)
            print('\r%{:3}['.format(process)+'**'*(process//10)+'->'+'..'*(10-process//10)+']',end='')
        for c in range(census.shape[1]):
            change=census[r,c]
            #weight to be determined
            if change<2:
                blend[r,c]=after[r+1,c+1]
            else:
                tmp=round((1.0*change*before[r+1,c+1]+(8.0-change)*after[r+1,c+1])/8)
                if tmp>255:
                    tmp=255
                elif tmp<0:
                    tmp=0
                blend[r,c]=tmp
    #print(cpa,cpb,blend)
    return blend

