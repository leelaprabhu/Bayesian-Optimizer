import math
import numpy
import random
import matlab.engine
eng = matlab.engine.start_matlab()
import numpy as np
import numpy.random as nprand
from sklearn import gaussian_process
import threading
from threading import Thread,Semaphore

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

l=125 #lambda
r=0.85
a=0.2
u=l/5
n=62
t=1.0/((2*(62**0.5))**0.5)
t2=1.0/((2*62)**0.5)

def islandBest(seed, Xa, Sa, thNo):
    minAll=100.0
    minParam=[]
    random.seed(seed)
    iter=0
    while(iter<500):
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
        print str(minIter)+" "+str(thNo)
        if(minAll>minIter):
            minAll=minIter
            minParam=parents[0][0]
        Xa=Xa2
        Sa=Sa2
        iter=iter+1
    print minAll
    print minParam
    return [minParam,Xa,Sa]

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
        global Xa
        global Sa
        global rPick
        [besta,X,S]=islandBest(rPick[i],Xa,Sa,i)
        best[i]=besta
        b1.wait(self.name)
        print str(best)+"~~~~~~~~"+str(i)
        X[-1]=best[0]
        X[-2]=best[1]
        X[-3]=best[2]
        X[-4]=best[3]
        b1.wait2(self.name)
        [besta,X,S]=islandBest(rPick[i],X,S,i)
        best[i]=besta
        b1.wait(self.name)
        print str(best)+"~~~~~~~~"+str(i)
        X[-1]=best[0]
        X[-2]=best[1]
        X[-3]=best[2]
        X[-4]=best[3]
        b1.wait2(self.name)
        [besta,X,S]=islandBest(rPick[i],X,S,i)
        best[i]=besta
        print str(best)+"~~~~~~~~"+str(i)
        X[-1]=best[0]
        X[-2]=best[1]
        X[-3]=best[2]
        X[-4]=best[3]
        b1.wait2(self.name)
        print '..... OUT .....'+str(i)

for i in range(noThreads):
    threads.append(myThread(i+1, "Thread-"+str(i+1)))

best=[[]]*4
rPick=[123,58,9,4666]
b1= Barrier(noThreads, 'b1')
def startAll():
    for i in range(noThreads):
        threads[i].start()

startAll()
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





