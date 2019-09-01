import pickle
import numpy as np

with open('filter230','rb') as fp:
    h=pickle.load(fp)

#print(h)
count=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#count=[0,0,0]
for ft in range(4):
    for a in range(24):
        for st in range(3):
            for co in range(3):
                if round(h[a,st,co,ft].sum(),3)!=0:
                    count[a]+=1

for i in range(24):
    print('angle',i,'filter学习度：%',(count[i]/(3*3*4))*100)

print('total filters学习度：%',(sum(count)/(24*3*3*4))*100)
