import math
import numpy
import random
import matlab.engine
eng = matlab.engine.start_matlab()
import numpy as np
import numpy.random as nprand
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import threading
from threading import Thread,Semaphore

def f1(Xa):
    param=list(Xa[:49])+[-3.5]*4+list(Xa[49:])
    param=map(lambda x: float(x), param)
    return eng.code2(1,param)

#Xi=np.concatenate((nprand.uniform(10.0,30.0,[2,6]),nprand.uniform(-0.3,0.3,[2,42]),nprand.uniform(0.0,30.0,[2,1]),nprand.uniform(-30.0,30.0,[2,1]),nprand.uniform(0.0,0.3,[2,6]),nprand.uniform(5.0,20.0,[2,6])),axis=1)
#yi = np.asarray([f1(Xi[0]),f1(Xi[1])])
l2=2
#gp = gaussian_process.GaussianProcess(corr='cubic',theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
lenSc=np.asarray([0.0]*62)
lenSc[0:6] =30.0 - 10.0
lenSc[6:48] =0.3 - (-0.3)
lenSc[48:49] =30.0 - 0.0
lenSc[49:50] =30.0 - (-30.0)
lenSc[50:56] =0.3 - 0.0
lenSc[56:62] =20.0 - 5.0

kernel = 34.4**2 * Matern(length_scale=lenSc)
gp = GaussianProcessRegressor(kernel=kernel)
Xi=np.concatenate((nprand.uniform(10.0,30.0,[l2,6]),nprand.uniform(-0.3,0.3,[l2,42]),nprand.uniform(0.0,30.0,[l2,1]),nprand.uniform(-30.0,30.0,[l2,1]),nprand.uniform(0.0,0.3,[l2,6]),nprand.uniform(5.0,20.0,[l2,6])),axis=1)
#Xi[0]= np.asarray([  1.96099393e+01,   1.58550894e+01,   1.09609545e+01,   1.22583452e+01, 1.13845861e+01,   2.32301699e+01,  -2.60330376e-01,  -9.59519233e-02, -3.92556934e-02,  -1.26138915e-03,  -3.75292346e-02,  -1.99628860e-01, 1.04272141e-01,  -2.98902972e-02,  -2.62179470e-02,  -3.41731807e-02, -2.90053980e-01,  -9.25959548e-02,   1.31797861e-01,  -8.80510331e-02, 4.03836475e-02,  -2.42839690e-01,  -1.26907174e-01,  -2.77364914e-01, 1.97789667e-01,  -2.65836378e-01,  -2.99287642e-01,  -8.78932771e-02, -3.57962043e-02,  -2.64148367e-01,   9.12228069e-02,  -5.01370722e-02, -3.07490603e-02,  -1.70573523e-01,   2.19795808e-02,  -2.28975755e-01, -2.36748681e-01,  -1.88787172e-01,  -2.50019760e-01,   5.18324371e-03, 2.37976187e-01,  -6.05562408e-02,  -5.76364679e-02,   2.95182555e-01, 2.03315798e-01,  -2.72488934e-01,   2.10069505e-02,  -2.50743497e-01, 2.66326081e+01,  -2.73519136e+01,   6.57654619e-02,   2.89214259e-01, 2.92997646e-01,   3.04660132e-01,   1.77162416e-01,   3.00612232e-01, 1.49886262e+01,   1.18078788e+01,   1.43817508e+01,   1.81800275e+01, 1.79311669e+01, 1.88175811e+01])
yi=map(lambda x: f1(x), Xi)
#gp = gaussian_process.GaussianProcess(corr='linear', theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
gp.fit(Xi, yi)
randomIter=nprand.randn()+2.0
paramB = np.asarray([  1.96099393e+01,   1.58550894e+01,   1.09609545e+01,   1.22583452e+01, 1.13845861e+01,   2.32301699e+01,  -2.60330376e-01,  -9.59519233e-02, -3.92556934e-02,  -1.26138915e-03,  -3.75292346e-02,  -1.99628860e-01, 1.04272141e-01,  -2.98902972e-02,  -2.62179470e-02,  -3.41731807e-02, -2.90053980e-01,  -9.25959548e-02,   1.31797861e-01,  -8.80510331e-02, 4.03836475e-02,  -2.42839690e-01,  -1.26907174e-01,  -2.77364914e-01, 1.97789667e-01,  -2.65836378e-01,  -2.99287642e-01,  -8.78932771e-02, -3.57962043e-02,  -2.64148367e-01,   9.12228069e-02,  -5.01370722e-02, -3.07490603e-02,  -1.70573523e-01,   2.19795808e-02,  -2.28975755e-01, -2.36748681e-01,  -1.88787172e-01,  -2.50019760e-01,   5.18324371e-03, 2.37976187e-01,  -6.05562408e-02,  -5.76364679e-02,   2.95182555e-01, 2.03315798e-01,  -2.72488934e-01,   2.10069505e-02,  -2.50743497e-01, 2.66326081e+01,  -2.73519136e+01,   6.57654619e-02,   2.89214259e-01, 2.92997646e-01,   3.04660132e-01,   1.77162416e-01,   3.00612232e-01, 1.49886262e+01,   1.18078788e+01,   1.43817508e+01,   1.81800275e+01, 1.79311669e+01, 1.88175811e+01])

def f(x):
    global gp
    global randomIter
    #y_pred, sigma2_pred = gp.predict(np.asarray(x).reshape(1,-1), eval_MSE=True)
    y_pred, sigma2_pred = gp.predict(np.asarray(x).reshape(1,-1), return_std=True)
    return list(y_pred-randomIter*(sigma2_pred**0.5))[0]

def fBulk(x):
    global gp
    global randomIter
    #y_pred, sigma2_pred = gp.predict(np.asarray(x), eval_MSE=True)
    y_pred, sigma2_pred = gp.predict(np.asarray(x), return_std=True)
    return np.asarray(map(lambda x,y: x-randomIter*(y**0.5),y_pred,sigma2_pred))


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

def gBulk(Xa):
    global n
    penalty=0.0
    can=np.zeros((Xa.shape[0],Xa.shape[1]),type(Xa[0][0]))
    can[:,0:6]=np.clip(Xa[:,0:6],10.0,30.0)
    can[:,6:48]=np.clip(Xa[:,6:48],-0.3,0.3)
    can[:,48:49]=np.clip(Xa[:,48:49],0.0,30.0)
    can[:,49:50]=np.clip(Xa[:,49:50],-30.0,30.0)
    can[:,50:56]=np.clip(Xa[:,50:56],0.0,0.3)
    can[:,56:62]=np.clip(Xa[:,56:62],5.0,20.0)
    penalty=(sum((can.T-Xa.T)**2.0)/n)**0.5
    return penalty

l=125 #lambda
r=0.85
a=0.2
u=l/5
n=62
t=1.0/((2*(62**0.5))**0.5)
t2=1.0/((2*62)**0.5)
itermain=0

def islandBest(seed, Xa, Sa, thNo):
    minAll=1000.0
    #minIterPrev=1000.0
    #minIter=200.0
    global itermain
    minIters=[2000.0,3000.0,4000.0,2500.0,3500.0,2000.0,3000.0,4000.0,2500.0,3500.0]
    minParam=Xa[0]
    random.seed(seed)
    iter=0
    if(itermain<100):
        iterTh=0
    else:
        iterTh=100
    while(((abs(max(minIters)-min(minIters))>0.001)or(iter<iterTh))and(iter<5000)):
        #print str(minAll)+"..........."+str(minAllPrev)
        indi=np.asarray(zip(Xa,Sa))
        fitf=fBulk(indi[:,0]) #np.asarray(map(lambda x: f(x),indi[:,0]))
        fitg=gBulk(indi[:,0]) #np.asarray(map(lambda x: g(x),indi[:,0]))
        fit=fitf+10000000.0*fitg
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
        minIters[1:]=minIters[0:-1]
        minIters[0]=f(parents[0][0])
        #print str(minIters[0])+" "+str(thNo)
        if(minAll>=minIters[0]):
            minAll=minIters[0]
            minParam=parents[0][0]
        #ret=[minParam,Xa,Sa]
        Xa=Xa2
        Sa=Sa2
        iter=iter+1
    print "----------------------------------"+str(iter)+"----------------------------------"
    ret=[minParam,Xa,Sa]
    #print minAll
    return ret

def islandBestLast(seed, Xa, Sa, thNo):
    minAll=1000.0
    #minIterPrev=1000.0
    #minIter=200.0
    global itermain
    minIters=[2000.0,3000.0,4000.0,2500.0,3500.0,2000.0,3000.0,4000.0,2500.0,3500.0,3500.0,3500.0,3500.0,3500.0,3500.0]
    minParam=Xa[0]
    random.seed(seed)
    iter=0
    if(itermain<500):
        iterTh=0
    else:
        iterTh=100
    while(((abs(max(minIters)-min(minIters))>0.001)or(iter<iterTh))and(iter<5000)):
    #while(iter<500):
        #print str(minAll)+"..........."+str(minAllPrev)
        indi=np.asarray(zip(Xa,Sa))
        fitf=fBulk(indi[:,0]) #np.asarray(map(lambda x: f(x),indi[:,0]))
        fitg=gBulk(indi[:,0]) #np.asarray(map(lambda x: g(x),indi[:,0]))
        fit=fitf+10000000.0*fitg
        X1=[x for (y,x) in sorted(zip(fit,indi), key=lambda pair: pair[0])]
        parents=X1[:u]
        Xa2=np.asarray([[0.0]*n]*l)
        Sa2=np.asarray([[0.0]*n]*l)
        Sa2t=np.asarray([[0.0]*n]*(l-u+1))
        randno1=nprand.randn((l-u+1),n)
        randno2=nprand.randn((l-u+1))
        randno3=nprand.randn((l-u+1),n)
#        rand4=np.exp((t2*randno2).T+(t*randno1).T)
#        Sa2[:k]
#        Xa2[:k]=map(lambda p1,p2: p1+r*(o-p2))
        for k in range(l):
            i=k%u
            if (k<(u-1)):
                p1 = parents[i][0]
                p2 = parents[i+1][0]
                o=parents[0][0]
                Xa2[k] = p1 + r*(o-p2)
                Sa2[k] = parents[i][1]
            else:
                #Sa2t[k-u+1]=parents[i][1]*np.exp(t2*randno2[k-u+1]+t*randno1[k-u+1])
                Sa2t[k-u+1]=parents[i][1]*np.exp(t2*randno2[k-u+1]+t*randno1[k-u+1])#rand4[k-u+1]
                Xa2[k]=parents[i][0]+Sa2t[k-u+1]*randno3[k-u+1]
                Sa2[k]=parents[i][1]+a*(Sa2t[k-u+1]-parents[i][1])
        minIters[1:]=minIters[0:-1]
        minIters[0]=f(parents[0][0])
        print str(minIters[0])+" "+str(thNo)
        if(minAll>=minIters[0]):
            minAll=minIters[0]
            minParam=parents[0][0]
        ret=[minParam,Xa,Sa]
        Xa=Xa2
        Sa=Sa2
        iter=iter+1
    print "----------------------------------"+str(iter)+"----------------------------------"
    #ret=[minParam,Xa,Sa]
    #print minAll
    return ret


Xa=np.concatenate((nprand.uniform(10.0,30.0,[l,6]),nprand.uniform(-0.3,0.3,[l,42]),nprand.uniform(0.0,30.0,[l,1]),nprand.uniform(-30.0,30.0,[l,1]),nprand.uniform(0.0,0.3,[l,6]),nprand.uniform(5.0,20.0,[l,6])),axis=1)

Sa=np.asarray([[0.0]*62]*l)
Sa[:,0:6] =30.0 - 10.0
Sa[:,6:48] =0.3 - (-0.3)
Sa[:,48:49] =30.0 - 0.0
Sa[:,49:50] =30.0 - (-30.0)
Sa[:,50:56] =0.3 - 0.0
Sa[:,56:62] =20.0 - 5.0
Sa=Sa/(62.0**0.5)
#----------------------------------------------
threads = []
noThreads=4

class Barrier:
    def __init__(self, n, name):
        self.n = n
        self.name = name
        self.count = 0
        self.mutex = Semaphore(1)
        self.turnstile = Semaphore(0)
        self.turnstile2 = Semaphore(1)
    
    def wait(self,caller):
        print self.name+' 1 '+caller
        self.mutex.acquire()
        self.count=self.count+1
        if (self.count==self.n):
            print 'Done----- '+self.name+' 1'
            self.turnstile2.acquire()
            self.turnstile.release()
        self.mutex.release()
        self.turnstile.acquire()
        self.turnstile.release()
    
    def wait2(self,caller):
        print self.name+' 2 '+caller
        self.mutex.acquire()
        self.count=self.count-1
        if (self.count==0):
            print 'Done----- '+self.name+' 2'
            self.turnstile.acquire()
            self.turnstile2.release()
        self.mutex.release()
        self.turnstile2.acquire()
        self.turnstile2.release()

class myThread (threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
    #self.x=[]
    def run(self):
        i=self.threadID-1
        global best
        Xa=np.concatenate((nprand.uniform(10.0,30.0,[l,6]),nprand.uniform(-0.3,0.3,[l,42]),nprand.uniform(0.0,30.0,[l,1]),nprand.uniform(-30.0,30.0,[l,1]),nprand.uniform(0.0,0.3,[l,6]),nprand.uniform(5.0,20.0,[l,6])),axis=1)
        global Sa
        global rPick
        while(1):
            [besta,X,S]=islandBestLast(rPick[i],Xa,Sa,i)
            best[i]=besta
            b1.wait(self.name)
            #print str(best)+"~~~~~~~~"+str(i)+"~~~~~~~~"+str(besta)
            X[-1]=best[0]
            X[-2]=best[1]
            X[-3]=best[2]
            X[-4]=best[3]
            b1.wait2(self.name)
            [besta,X,S]=islandBestLast(rPick[i],X,S,i)
            best[i]=besta
            b1.wait(self.name)
            #print str(best)+"~~~~~~~~"+str(i)
            X[-1]=best[0]
            X[-2]=best[1]
            X[-3]=best[2]
            X[-4]=best[3]
            b1.wait2(self.name)
            [besta,X,S]=islandBestLast(rPick[i],X,S,i)
            best[i]=besta
            b1.wait(self.name)
            #print str(best)+"~~~~~~~~"+str(i)
            X[-1]=best[0]
            X[-2]=best[1]
            X[-3]=best[2]
            X[-4]=best[3]
            b1.wait2(self.name)
            b2.wait(self.name)
            b2.wait2(self.name)
            print '..... OUT .....'+str(i)

for i in range(noThreads):
    threads.append(myThread(i+1, "Thread-"+str(i+1)))

best=[[]]*4
rPick=[123,58,9,4666]
b1= Barrier(noThreads, 'b1')
b2= Barrier(noThreads+1, 'b2')

def startAll():
    for i in range(noThreads):
        threads[i].start()

startAll()
while(1):
    b2.wait("Main")
#    if(itermain<200):
#        randomIter=1.0
#    else:
#        randomIter=3.0
    randomIter=10.0#*(min(yi)/60.0)/
    argminx=np.argmin([f(best[0]),f(best[1]),f(best[2]),f(best[3])])
    cc=best[argminx]
    nv=f1(cc)
    while(nv>300):#or((nv>65)and(itermain<10)):
        print "----------redo-----------"+str(nv)
        cc=np.concatenate((nprand.uniform(10.0,30.0,[1,6]),nprand.uniform(-0.3,0.3,[1,42]),nprand.uniform(0.0,30.0,[1,1]),nprand.uniform(-30.0,30.0,[1,1]),nprand.uniform(0.0,0.3,[1,6]),nprand.uniform(5.0,20.0,[1,6])),axis=1)
        nv=f1(cc[0])
    #    if(argminx==0):
    #        cc1=best[np.argmin([f(best[1]),f(best[2]),f(best[3])])]
    #        nv1=f1(cc1)
    #    elif(argminx==1):
    #        cc1=best[np.argmin([f(best[1]),f(best[2]),f(best[3])])]
    #        nv1=f1(cc1)
    #    elif(argminx==2):
    #        cc1=best[np.argmin([f(best[1]),f(best[2]),f(best[3])])]
    #        nv1=f1(cc1)
    #    else:
    #        cc1=best[np.argmin([f(best[1]),f(best[2]),f(best[3])])]
    #        nv1=f1(cc1)
    Xi=numpy.row_stack((Xi,cc))
    if(len(Xi)>7):
        print "++++++"+str(sum((sum(((Xi[-6:-2]-Xi[-5:-1])**2.0).T)/62.0)**0.5/4.0))+"++++++"
        #randomIter=5.0*(min(yi)/60.0)/(sum((sum(((Xi[-6:-2]-Xi[-5:-1])**2.0).T)/62.0)**0.5/4.0))
    yi=np.concatenate((yi,[nv]))
    print Xi
    print yi
    y_predB, sigma2_predB = gp.predict(np.asarray(paramB).reshape(1,-1), return_std=True)
    valB=list(y_predB-randomIter*(sigma2_predB**0.5))[0]
    y_predS, sigma2_predS = gp.predict(np.asarray(Xi[-1]).reshape(1,-1), return_std=True)
    valS=list(y_predS-randomIter*(sigma2_predS**0.5))[0]
    print "=========="+str(min(yi))+"=========="+str(itermain)+"==========="+str(y_predB)+" "+str(sigma2_predB**0.5)+" "+str(valB)+" "+str(y_predS)+"----"+str(sigma2_predS**0.5)+" "+str(valS)
    try:
        gp.fit(Xi,yi)
    except:
        None
    b2.wait2("Main")
    itermain=itermain+1

for thr in threads:
    thr.join()

#[best1,Xa1,Sa1]=islandBest(123,Xa,Sa)
#[best2,Xa2,Sa2]=islandBest(58,Xa,Sa)
#[best3,Xa3,Sa3]=islandBest(9,Xa,Sa)
#[best4,Xa4,Sa4]=islandBest(4666,Xa,Sa)
#
#Xa1[-1]=best1
#Xa1[-2]=best2
#Xa1[-3]=best3
#Xa1[-4]=best4
#
#Xa2[-1]=best1
#Xa2[-2]=best2
#Xa2[-3]=best3
#Xa2[-4]=best4
#
#Xa3[-1]=best1
#Xa3[-2]=best2
#Xa3[-3]=best3
#Xa3[-4]=best4
#
#Xa4[-1]=best1
#Xa4[-2]=best2
#Xa4[-3]=best3
#Xa4[-4]=best4
#
#[best1,Xa1,Sa1]=islandBest(123,Xa1,Sa1)
#[best2,Xa2,Sa2]=islandBest(58,Xa2,Sa2)
#[best3,Xa3,Sa3]=islandBest(9,Xa3,Sa3)
#[best4,Xa4,Sa4]=islandBest(4666,Xa4,Sa4)
#
#
#Xa1[-1]=best1
#Xa1[-2]=best2
#Xa1[-3]=best3
#Xa1[-4]=best4
#
#Xa2[-1]=best1
#Xa2[-2]=best2
#Xa2[-3]=best3
#Xa2[-4]=best4
#
#Xa3[-1]=best1
#Xa3[-2]=best2
#Xa3[-3]=best3
#Xa3[-4]=best4
#
#Xa4[-1]=best1
#Xa4[-2]=best2
#Xa4[-3]=best3
#Xa4[-4]=best4
#
#
#[best1,Xa1,Sa1]=islandBest(123,Xa1,Sa1)
#[best2,Xa2,Sa2]=islandBest(58,Xa2,Sa2)
#[best3,Xa3,Sa3]=islandBest(9,Xa3,Sa3)
#[best4,Xa4,Sa4]=islandBest(4666,Xa4,Sa4)





