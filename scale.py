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

#def f1(param):
#    param=map(lambda x: float(x), param)
#    return eng.codelv(1,param)
dim=40
tar=nprand.uniform(0.0,10.0,dim)

def f1(param):
    global tar
    return sum((param-tar)**2.0)/len(param)

Xi=nprand.uniform(0.0,10.0,[2,dim])
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
    can=np.asarray([0.0]*dim)
    can=np.clip(Xa,0.0,10.0)
    penalty=(sum((can-Xa)**2.0)/n)**0.5
    return penalty

l=125 #lambda
r=0.85
a=0.2
u=l/5
n=dim
t=1.0/((2*(n**0.5))**0.5)
t2=1.0/((2*n)**0.5)

def islandBest(seed, Xa, Sa, thNo):
    minAll=1000.0
    #minIterPrev=1000.0
    #minIter=200.0
    minIters=[200.0,300.0,400.0,250.0,350.0]
    minParam=Xa[0]
    random.seed(seed)
    iter=0
    while((abs(max(minIters)-min(minIters))>0.01)and(iter<500)):
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
        Xa=Xa2
        Sa=Sa2
        iter=iter+1
    #print minAll
    #print minParam
    return [minParam,Xa,Sa]

Xa=nprand.uniform(0.0,10.0,[l,dim])

Sa=np.asarray([[10.0]*dim]*l)
Sa=Sa/(dim**0.5)
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
            [besta,X,S]=islandBest(rPick[i],X,S,i)
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
itermain=0
while(1):
    b2.wait("Main")
    randomIter=1.0
    cc=best[np.argmin([f(best[0]),f(best[1]),f(best[2]),f(best[3])])]
    nv=f1(cc)
    Xi=numpy.row_stack((Xi,cc))
    if(len(Xi)>7):
        print "++++++"+str(sum((sum((Xi[-6:-2]-Xi[-5:-1])**2.0)/4.0)**0.5)/len(Xi[0]))+"++++++"
    yi=np.concatenate((yi,[nv]))
    print Xi
    print tar
    print yi
    try:
        gp.fit(Xi,yi)
    except:
        None
    b2.wait2("Main")
    itermain=itermain+1

for thr in threads:
    thr.join()





