
# coding: utf-8

# In[1]:

from scipy.integrate import odeint
import numpy as np
from matplotlib.pylab import *
import matplotlib.pylab as pylab
from numpy import genfromtxt
import math
import sys
num=47
#print paramR=param[0:6];
#T=param[6:42].reshape([6,6]);
#m=param[42:48];
#h=param[48:54];
#D=param[54:60];
#t1_2=param[60:66];#print t1_2
# In[2]:

bcd14a = genfromtxt('bcd14aj12.csv', delimiter=',')
v34 = genfromtxt('v34j12.csv', delimiter=',')
v93 = genfromtxt('v93j12.csv', delimiter=',')


# In[3]:

x14a = genfromtxt('x14aj12.csv', delimiter=',')
x14a = x14a.reshape([8,58,6])
x14a = np.swapaxes(x14a,1,2)
init14a=x14a[0,:,:].reshape(348)


# print init14a

# In[4]:

times=np.asarray([24.2250,30.4750,36.7250,42.9750,49.2250,55.4750,61.7250,67.9750]);


# In[5]:

def cycle14a(g2,t,aa,param):
    R=param[0:6];
    T=param[6:42].reshape([6,6]);
    m=param[42:48];
    h=param[48:54];
    D=param[54:60];
    t1_2=param[60:66];
    tp=argmin(abs(t-times))
    dg=np.asarray([[0.0]*58]*6)
    g=g2.reshape([6,58]);
    for a in range(6): #gene
        for i in range(58): #nucleus
            Tv=T[a][0]*g[0][i]+T[a][1]*g[1][i]+T[a][2]*g[2][i]+T[a][3]*g[3][i]+T[a][4]*g[4][i]+T[a][5]*g[5][i]
            ua=Tv+m[a]*bcd14a[i]+h[a]
            if(i==0):
                vv=(v34[tp][a]-g[a][i])+(g[a][i+1]-g[a][i])
            elif(i==57):
                vv=(g[a][i-1]-g[a][i])+(v93[tp][a]-g[a][i])
            else:
                vv=(g[a][i-1]-g[a][i])+(g[a][i+1]-g[a][i])
            lmbd=math.log(2.0)/t1_2[a]
            dg[a][i]=R[a]*0.5*(ua/((ua**2.0+1)**0.5)+1)+D[a]*vv-lmbd*g[a][i]
    dg2=list(dg.reshape(58*6));
    return dg2

#y3=odeint(cycle14a,init14a,[24.225, 30.475, 36.725, 42.975, 49.225, 55.475, 61.725, 67.975, 71.1])
# In[6]:

def plot(res,x,lb,ub):
    c=['b','r','m','y','k','g']
    for i in range(6):
        plt.plot(range(lb,ub),x[i,:],c[i]+'-')
        plt.plot(range(lb,ub),res[i,:],c[i]+'--')


# In[7]:

def plotRes():
    for i in range(8):
        yres=y3[i,:].reshape([6,58])
        plot(yres,x14a[i,:],35,93)
        plt.show()


# In[8]:

def calcErr(y3):
    rmse=0.0
    for i in range(8):
        yy=x14a[i,:,:].reshape(348)
        rmse=rmse+sum((yy-y3[i,:])**2.0)
    rmse=(rmse/(6*58*8))**0.5
    return rmse


# In[9]:

def odefunc(param):
    y3=odeint(cycle14a,init14a,times, args=(1,param))
    return calcErr(y3)


# In[10]:

param = genfromtxt('param.csv', delimiter=',')
#odefunc(param)
#num=10
print str(num)
min=100000
mval=0
t=np.linspace(-0.3,0.3,200)
for i in t:
    param[num]=i
    c=odefunc(param)
    if(c<min):
        min=c
        mval=i
        
print mval
# In[ ]:



