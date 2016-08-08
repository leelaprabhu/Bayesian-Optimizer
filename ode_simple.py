from scipy.integrate import odeint
import numpy as np
from matplotlib.pylab import *
import matplotlib.pylab as pylab
from numpy import genfromtxt
#R=[20.0, 19.608, 16.373, 15.789, 12.185, 11.906]
#T=[[-0.068, -0.073, -0.050, -0.056, -0.038, -0.034],
#   [0.022, 0.019, 0.001, 0.011, -0.166, 0.003],
#   [0.033, -0.014, 0.017, -0.076, -0.015, -0.080],
#   [0.029, -0.018, -0.110, 0.011, -0.001, -0.020],
#   [0.037, -0.027, -0.024, -0.090, 0.045, -0.077],
#   [-0.018, -0.106, -0.106, -0.082, -0.137, -0.003]]
#m=[-0.040, 0.050, 0.129, 0.177, 0.097, -0.007]
#h=[13.459, -3.500, -3.500, -3.500, -3.500, 8.173]
#D=[0.200, 0.200, 0.200, 0.142, 0.200, 0.200]
#t1_2=[18.000, 7.254, 8.980, 9.577, 12.499, 16.842]

param = genfromtxt('param.csv', delimiter=',')
R=param[0:6];
T=param[6:42].reshape([6,6]);
m=param[42:48];
h=param[48:54];
D=param[54:60];
t1_2=param[60:66];
v16=[78.77,    104.27,    28.35,    43.22,    0.0,    20.69];
v47=[104.26,    7.56,       15.42,    40.34,    0.0,    26.88];
init13 = genfromtxt('init13.csv', delimiter=',')
bcd13 = genfromtxt('bcd13.csv', delimiter=',')
def cycle13(g2,t):
    global v16
    global v34
    global R
    global T
    global m
    global h
    global D
    global t1_2
    dg=np.asarray([[0.0]*30]*6)
    g=g2.reshape([6,30]);
    for a in range(6): #gene
        for i in range(30): #nucleus
            Tv=T[a][0]*g[0][i]+T[a][1]*g[1][i]+T[a][2]*g[2][i]+T[a][3]*g[3][i]+T[a][4]*g[4][i]+T[a][5]*g[5][i]
            ua=Tv+m[a]*bcd13[i]+h[a]
            if(i==0):
                vv=(v16[a]-g[a][i])+(g[a][i+1]-g[a][i])
            elif(i==29):
                vv=(g[a][i-1]-g[a][i])+(v47[a]-g[a][i])
            else:
                vv=(g[a][i-1]-g[a][i])+(g[a][i+1]-g[a][i])
            lmbd=2.0/t1_2[a]
            dg[a][i]=R[a]+0.5*(ua/((ua**2.0+1)**0.5)+1)+D[a]*vv-lmbd*g[a][i]
    dg2=list(dg.reshape(30*6));
    return dg2

x13 = genfromtxt('x13.csv', delimiter=',').T
res=x13.reshape(180)
y=odeint(cycle13,init13,[0.0, 10.55, 16.0])
yres=y[1,:].reshape([6,30])
#print (sum((res-y[1,:])**2.0)/180.0)**0.5
print x13
exit(0);
#v34=[72.18, 173.87, 31.06, 77.98, 33.60, 0.99]
#v93=[113.31, 8.99, 14.99, 39.54, 20.75, 77.15]
#xyz=[cad,hb,Kr,gt,kni,tll]
#bcd= np.load('Data/bicoid.npy')[0]

def cycle14a(g2,t):
    dg=[0.0]*58*6
    g=g2.reshape([6,58]);
    for a in range(6): #gene
        for i in range(58): #nucleus
            Tv=T[a][0]*g[0][i]+T[a][1]*g[1][i]+T[a][2]*g[2][i]+T[a][3]*g[3][i]+T[a][4]*g[4][i]+T[a][5]*g[5][i]
            ua=Tv+m[a]*bcd14a[i]+h[a]
            if(i==0):
                vv=(v34[a]-g[a][i])+(g[a][i+1]-g[a][i])
            elif(i==57):
                vv=(g[a][i-1]-g[a][i])+(v93[a]-g[a][i])
            else:
                vv=(g[a][i-1]-g[a][i])+(g[a][i+1]-g[a][i])
            lmbd=2.0/t1_2[a]
            dg[a][i]=R[a]+0.5*(ua/((ua**2.0+1)**0.5)+1)+D[a]*vv-lmbd*g[a][i]
    dg2=dg.reshape(58*6);
    return dg2

def mitosis(g2,t):
    dg=[0.0]*30*6
    g=g2.reshape([6,30]);
    for a in range(6): #gene
        for i in range(30): #nucleus
            Tv=T[a][0]*g[0][i]+T[a][1]*g[1][i]+T[a][2]*g[2][i]+T[a][3]*g[3][i]+T[a][4]*g[4][i]+T[a][5]*g[5][i]
            ua=Tv+m[a]*bcd13[i]+h[a]
            if(i==0):
                vv=(v16[a]-g[a][i])+(g[a][i+1]-g[a][i])
            elif(i==29):
                vv=(g[a][i-1]-g[a][i])+(v47[a]-g[a][i])
            else:
                vv=(g[a][i-1]-g[a][i])+(g[a][i+1]-g[a][i])
            lmbd=2.0/t1_2[a]
            dg[a][i]=R[a]+0.5*(ua/((ua**2.0+1)**0.5)+1)+D[a]*vv-lmbd*g[a][i]
    dg2=dg.reshape(30*6);
    return dg2

############
#read x13
#read init13
#load bicoid13

time=np.linspace(22.0,70.0,100)
init=np.load('Data/init0.npy')
yinit=init
y=odeint(cycle13,init13,[0.0, 10.55, 16.0])

y2=odeint(mitosis,y[3,:],[16.0, 21.1])

y3=odeint(cycle14a,y[3,:],[21.1, 24.225, 30.475, 36.725, 42.975, 49.225, 55.475, 61.725, 67.975])
print y[:,0]