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

def f1(param):
    param=map(lambda x: float(x), param)
    return eng.codelv(1,param)

#def f1(param):
#    a=np.asarray([1.0,0.05,0.02,0.5])
#    return (sum((param-a)**2.0)/len(param))**0.5

Xi=np.concatenate((nprand.uniform(0.0,2.0,[2,1]),nprand.uniform(0.0,0.1,[2,1]),nprand.uniform(0.0,0.05,[2,1]),nprand.uniform(0.0,1.0,[2,1])),axis=1)
yi = np.asarray([f1(Xi[0]),f1(Xi[1])])
gp = gaussian_process.GaussianProcess(corr='linear', theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
gp.fit(Xi, yi)
randomIter=nprand.randn()+2.0
def f(x):
    global gp
    global randomIter
    y_pred, sigma2_pred = gp.predict(np.asarray(x).reshape(1,-1), eval_MSE=True)
    return list(y_pred-randomIter*(sigma2_pred**0.5))[0]

def g(Xa):
    global n
    penalty=0.0
    can=np.asarray([0.0]*4)
    can[0]=np.clip(Xa[0],0.0,2.0)
    can[1]=np.clip(Xa[1],0.0,0.1)
    can[2]=np.clip(Xa[2],0.0,0.05)
    can[3]=np.clip(Xa[3],0.0,1.0)
    penalty=(sum((can-Xa)**2.0)/n)**0.5
    return penalty

l=125 #lambda
r=0.85
a=0.2
u=l/5
n=4
t=1.0/((2*(4**0.5))**0.5)
t2=1.0/((2*4)**0.5)
itermain=0

def islandBest(seed, Xa, Sa, thNo):
    minAll=1000.0
    #minIterPrev=1000.0
    #minIter=200.0
    global itermain
    minIters=[200.0,300.0,400.0,250.0,350.0,200.0,300.0,400.0,250.0,350.0]
    minParam=Xa[0]
    random.seed(seed)
    iter=0
    if(itermain<100):
        iterTh=0
    else:
        iterTh=100
    while(((abs(max(minIters)-min(minIters))>0.001)or(iter<iterTh))and(iter<500)):
        #print str(minAll)+"..........."+str(minAllPrev)
        indi=np.asarray(zip(Xa,Sa))
        fitf=np.asarray(map(lambda x: f(x),indi[:,0]))
        fitg=np.asarray(map(lambda x: g(x),indi[:,0]))
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
        print str(minIters[0])+" "+str(thNo)
        if(minAll>=minIters[0]):
            minAll=minIters[0]
            minParam=parents[0][0]
        #ret=[minParam,Xa,Sa]
        Xa=Xa2
        Sa=Sa2
        iter=iter+1
    ret=[minParam,Xa,Sa]
    #print minAll
    return ret

def islandBestLast(seed, Xa, Sa, thNo):
    minAll=1000.0
    #minIterPrev=1000.0
    #minIter=200.0
    global itermain
    minIters=[200.0,300.0,400.0,250.0,350.0,200.0,300.0,400.0,250.0,350.0]
    minParam=Xa[0]
    random.seed(seed)
    iter=0
    if(itermain<500):
        iterTh=0
    else:
        iterTh=100
    while(((abs(max(minIters)-min(minIters))>0.001)or(iter<iterTh))and(iter<500)):
        #print str(minAll)+"..........."+str(minAllPrev)
        indi=np.asarray(zip(Xa,Sa))
        fitf=np.asarray(map(lambda x: f(x),indi[:,0]))
        fitg=np.asarray(map(lambda x: g(x),indi[:,0]))
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
        print str(minIters[0])+" "+str(thNo)
        if(minAll>=minIters[0]):
            minAll=minIters[0]
            minParam=parents[0][0]
        ret=[minParam,Xa,Sa]
        Xa=Xa2
        Sa=Sa2
        iter=iter+1
        #ret=[minParam,Xa,Sa]
    #print minAll
    return ret


Xa=np.concatenate((nprand.uniform(0.0,2.0,[l,1]),nprand.uniform(0.0,0.1,[l,1]),nprand.uniform(0.0,0.05,[l,1]),nprand.uniform(0.0,1.0,[l,1])),axis=1)

Sa=np.asarray([[0.0]*4]*l)
Sa[:,0] =2.0
Sa[:,1] =0.1
Sa[:,2] =0.05
Sa[:,3] =1.0
Sa=Sa/(4.0**0.5)
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
        while(1):
            Xa=np.concatenate((nprand.uniform(0.0,2.0,[l,1]),nprand.uniform(0.0,0.1,[l,1]),nprand.uniform(0.0,0.05,[l,1]),nprand.uniform(0.0,1.0,[l,1])),axis=1)
            [besta,X,S]=islandBest(rPick[i],Xa,Sa,i)
            best[i]=besta
            b1.wait(self.name)
            #print str(best)+"~~~~~~~~"+str(i)+"~~~~~~~~"+str(besta)
            X[-1]=best[0]
            X[-2]=best[1]
            X[-3]=best[2]
            X[-4]=best[3]
            b1.wait2(self.name)
            [besta,X,S]=islandBest(rPick[i],X,S,i)
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

best=[[]]*noThreads
rPick=[123,58,9,4666]
b1= Barrier(noThreads, 'b1')
b2= Barrier(noThreads+1, 'b2')

def startAll():
    for i in range(noThreads):
        threads[i].start()

startAll()
while(1):
    b2.wait("Main")
    randomIter=5.0*100.0/(100.0+itermain)
    argminx=np.argmin([f(best[0]),f(best[1]),f(best[2]),f(best[3])])
    cc=best[argminx]
    nv=f1(cc)
    while(nv>100):
        print "----------redo-----------"+str(nv)
        cc=np.concatenate((nprand.uniform(0.0,2.0,[1,1]),nprand.uniform(0.0,0.1,[1,1]),nprand.uniform(0.0,0.05,[1,1]),nprand.uniform(0.0,1.0,[1,1])),axis=1)[0]
        nv=f1(cc)
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
        print "++++++"+str(sum((sum((Xi[-6:-2]-Xi[-5:-1])**2.0)/4.0)**0.5)/4.0)+"++++++"
    yi=np.concatenate((yi,[nv]))
    print Xi
    print yi
    print "======================"+str(min(yi))+"======================"+str(itermain)+"======================"+str(f(np.asarray([1.0,0.05,0.02,0.5])))+"======================"
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





