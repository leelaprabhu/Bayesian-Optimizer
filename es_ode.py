import math
import numpy
import random
import matlab.engine
eng = matlab.engine.start_matlab()
import numpy as np
import numpy.random as nprand
from sklearn import gaussian_process

def f(Xa):
    param=list(Xa[:49])+[-3.5]*4+list(Xa[49:])
    param=map(lambda x: float(x), param)
    return eng.code2(1,param)

def g(Xa):
    global n
    penalty=0.0
    can=np.asarray([0.0]*62)
    can[0:6]=np.clip(Xa[0:6],10.0,30.0)
    can[6:48]=np.clip(Xa[6:48],-0.3,0.3)
    can[48:49]=np.clip(Xa[48:49],0.0,30.0)
    can[49:50]=np.clip(Xa[49:50],-30.0,30.0)
    can[50:56]=np.clip(Xa[50:56],0.0,0.3)
    can[56:62]=np.clip(Xa[56:62],5.0,20.0)
    penalty=(sum((can-Xa)**2.0)/n)**0.5
    return penalty

l=200 #lambda
r=0.85
a=0.2
u=l/5
n=62
t=1.0/((2*(62**0.5))**0.5)
t2=1.0/((2*62)**0.5)

Xa=np.concatenate((nprand.uniform(10.0,30.0,[l,6]),nprand.uniform(-0.3,0.3,[l,42]),nprand.uniform(0.0,30.0,[l,1]),nprand.uniform(-30.0,30.0,[l,1]),nprand.uniform(0.0,0.3,[l,6]),nprand.uniform(5.0,20.0,[l,6])),axis=1)

Sa=np.asarray([[0.0]*62]*l)
Sa[:,0:6] =30.0 - 10.0
Sa[:,6:48] =0.3 - (-0.3)
Sa[:,48:49] =30.0 - 0.0
Sa[:,49:50] =30.0 - (-30.0)
Sa[:,50:56] =0.3 - 0.0
Sa[:,56:62] =20.0 - 5.0
Sa=Sa/(62.0**0.5)

minAll=100.0

while(1):
    indi=np.asarray(zip(Xa,Sa))
    fitf=np.asarray(map(lambda x: f(x),indi[:,0]))
    fitg=np.asarray(map(lambda x: g(x),indi[:,0]))
    fit=fitf+100.0*fitg
    X1=[x for (y,x) in sorted(zip(fit,indi), key=lambda pair: pair[0])]
    parents=X1[:u]
    Xa2=np.asarray([[0.0]*n]*l)
    Sa2=np.asarray([[0.0]*n]*l)
    Sa2t=np.asarray([[0.0]*n]*(l-u+1))
    randno1=nprand.randn((l-u+1),n)
    randno2=nprand.randn((l-u+1))
    randno3=nprand.randn((l-u+1),n)
    for k in range(l):
        i=k%u
        if (k<(u-1)):
            p1 = parents[i][0]
            p2 = parents[i+1][0]
            o=parents[0][0]
            Xa2[k] = p1 + r*(o-p2)
            Sa2[k] = parents[i][1]
        else:
            Sa2t[k-u+1]=parents[i][1]*np.exp(t2*randno2[k-u+1]+t*randno1[k-u+1])
            Xa2[k]=parents[i][0]+Sa2t[k-u+1]*randno3[k-u+1]
            Sa2[k]=parents[i][1]+a*(Sa2t[k-u+1]-parents[i][1])
    minIter=f(parents[0][0])
    #print str(minIter)
    if(minAll>minIter):
        minAll=minIter
        minParam=parents[0][0]
    print minIter
    print minParam
    Xa=Xa2
    Sa=Sa2






