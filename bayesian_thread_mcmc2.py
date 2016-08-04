#!/usr/bin/python
import numpy 
import numpy as np
from sklearn import gaussian_process
import threading
import time
from threading import Thread,Semaphore
import random
import numpy.random as nprand
from math import *

def acq(y_pred,sigma2_pred,j):
    return map(lambda a,b: a-(b**0.5),y_pred,sigma2_pred)

#def f1(x):
#    return (x[0]-5.0)**2.0+(x[1]-5.0)**2.0+(x[2]-5.0)**2.0+(x[3]-5.0)**2.0+(x[4]-5.0)**2.0

def f1(x):
	global dim
	a=[5.0]*dim
	z=(x[0]-a[0])**2.0
	for i in range(1,len(x)):
		z=z+(x[i]-a[i])**2.0    	
	return z

def f(x):
	global dim
	a=[5.0]*dim
	z=(x[:,0]-a[0])**2.0
    	for i in range(1,len(x[0])):
		z=z+(x[:,i]-a[i])**2.0
	return z
#def f(x):
    #return (x[:,0]-5.0)**2.0+(x[:,1]-5.0)**2.0+(x[:,2]-5.0)**2.0+(x[:,3]-5.0)**2.0+(x[:,4]-5.0)**2.0

#X=np.atleast_2d([[8.0,0.0,4.0,6.0,9.0,1.0,7.5,9.8,4.2,0.8],[1.0, 9.0, 2.0,7.0,3.0,6.0,2.3,8.2,3.9,7.0]])
dim=66
X=nprand.uniform(low=0.0, high=10.0, size=[2,dim])
maxprev=X[0]
prevmax1=f1(X[0])
y = f(X).ravel()
#x1=[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
x1=np.linspace(0,10,100)
#v1,v2,v3,v4,v5= np.meshgrid(x1,x1,x1,x1,x1[0])
gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
#n=10
noThreads=10
x_t=0
thres=0.02
limit=400
neigh=100
#x=np.column_stack((xv.flatten().T,yv.flatten().T,zv.flatten().T,wv.flatten().T))
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

def sdnorm(p):
	global gp
	#y_pred, sigma2_pred = gp.predict(p.reshape(1,-1), eval_MSE=True)
	#ac= y_pred-(sigma2_pred**0.5)
	y_pred, sigma2_pred = gp.predict(p, eval_MSE=True)
	ac= map(lambda a,b: a-(b**0.5),y_pred,sigma2_pred)
	return ac

def sdnorm1(p):
	global gp
	y_pred, sigma2_pred = gp.predict(p.reshape(1,-1), eval_MSE=True)
	ac= y_pred-(sigma2_pred**0.5)
	#y_pred, sigma2_pred = gp.predict(p, eval_MSE=True)
	#ac= map(lambda a,b: a-(b**0.5),y_pred,sigma2_pred)
	return ac

def getSamples1():
	n = 1000
	alpha = 0.1
	global dim
	#x = [0.]*dim
	vec = []
	resl= []
	#vec.append(x)
	iter=1
	for j in range(iter):
    		x = nprand.uniform(low=0.0, high=10.0, size=[1,dim])[0]
            vec.append(x)
            resl.append(sdnorm(x))
            noise=nprand.uniform(low=-alpha, high=alpha, size=[n,dim])
    		for i in xrange(1,n):
        		can = x + noise[i] #candidate
                k=sdnorm(can)
        		aprob = min([1.,resl[-1]/k]) #acceptance probability
        		u = uniform(0,1)
        		if u < aprob:
                        x = can
            			vec.append(x)
                        resl.append(k)
    		#print "done"
    		#print len(vec)/(n*(iter+1.0))
	return [vec,resl]

def getSamples():
	n = 1000
	alpha = 0.1
	global dim
	#x = [0.]*dim
	vec = []
	resl= []
	#vec.append(x)
	iter=100
	#for j in range(iter)
    x = nprand.uniform(low=0.0, high=10.0, size=[iter,dim])
	vec.extend(x)
	resl.extend(sdnorm(x))
	noise=nprand.uniform(low=-alpha, high=alpha, size=[n,dim])
    for i in xrange(1,n):
        	can = x + noise[i] #candidate
            can=np.clip(can,0.0,10)
            #can=[[max(min(u,10.0),0.0) for u in yy] for yy in can]
            #can=map(lambda x: map(lambda y: max(0.0,min(10.0,y)), x), can)
            #can=map(lambda x: map(lambda y: max(0.0,y), x), can)
            k=sdnorm(can)
            k2=sdnorm(x)
        	aprob = map(lambda a,b: min([1.,a/b]),k2,k) #acceptance probability
        	u = nprand.uniform(0,1,iter)
            for j in range(iter):
        		if u[j] < aprob[j]:
            			x[j] = can[j]
            			vec.append(x[j])
                        resl.append(k[j])
	return [vec,resl]

class myThread (threading.Thread):
    	def __init__(self, threadID, name):
        	threading.Thread.__init__(self)
        	self.threadID = threadID
        	self.name = name
            #self.x=[]
    	def run(self):
            m=10
            #m=1
            i=self.threadID-1
            for l in range(1):
                b3.wait(self.name)
                #self.x= np.column_stack((v1[i*m:(i+1)*m].flatten().T,v2[i*m:(i+1)*m].flatten().T,v3[i*m:(i+1)*m].flatten().T,v4[i*m:(i+1)*m].flatten().T,v5[i*m:(i+1)*m].flatten().T))
                #self.x=nprand.uniform(low=0.0, high=10.0, size=[10000000,6])
                #print "%s: %s %s" % (self.name, time.ctime(time.time()), self.x[-1])
                b3.wait2(self.name)
                global check
                global breakAll
                for j in range(limit):
                    #if(breakAll):
                    #	break
                    b.wait(self.name)
                    b.wait2(self.name)
                    if(i<noThreads/2):
                        b1.wait(self.name)
                        #print '======= '+str(check)
                        self.run2(j)
                        b1.wait2(self.name)
                        b2.wait(self.name)
                        b2.wait2(self.name)
                    else:
                        b1.wait(self.name)
                        b1.wait2(self.name)
                        b2.wait(self.name)
                        #print '======= '+str(check)
                        self.run2(j)
                        b2.wait2(self.name)
                    b4.wait(self.name)
                    if(breakAll):
                        print 'BREAK!!!'
                        break
                    else:
                        b4.wait2(self.name)
			print self.name+'----- OUT -----'+str(i)
			b4.wait2(self.name)
		print '..... OUT .....'+str(i) 
 
			#print '..... '+str(i)

	def run2(self,j):    	
		#y_pred, sigma2_pred = gp.predict(self.x, eval_MSE=True)
        	#ac=acq(y_pred, sigma2_pred)
		#minpos[self.threadID-1]=numpy.argmin(ac)
		#minParam[self.threadID-1]=self.x[numpy.argmin(ac)]
		#minval[self.threadID-1]=ac[minpos[self.threadID-1]]
		min1=10000000
		for i in range(1):
        		#print i
			global maxprev
			#x=nprand.uniform(low=0.0, high=10.0, size=[500000,10])	
			x2=maxprev+nprand.uniform(low=-0.1, high=0.1,size=[100,dim])
			#x[0:100]=xx
        		#y_pred, sigma2_pred = gp.predict(self.x[i*1000000:(i+1)*1000000], eval_MSE=True)
			[x,ac]=getSamples()			
			y_pred2, sigma2_pred2 = gp.predict(x2, eval_MSE=True)
			#y_pred, sigma2_pred = gp.predict(self.x[i*100:(i+1)*100], eval_MSE=True)
            ac2=acq(y_pred2, sigma2_pred2,j)
			#print 'DIMENSION '+str(len(ac2))
			#print 'DIMENSION '+str(len(ac))
			if(j>neigh):			
				x=np.concatenate((x,x2))
				ac=ac+ac2
			#print 'DIMENSION '+str(len(ac))
            x_t1=numpy.argmin(ac)
			#print str(self.x[i*1000000+x_t1])+'========'+str(self.x[i*1000000])+'======='+str(self.x[(i+1)*1000000-1])
            minx=ac[x_t1]
            if(minx<min1):
                min1=minx
				#minv=self.x[i*1000000+x_t1]
				minv=x[x_t1]
				#miny=y_pred[x_t1]
				#mins=sigma2_pred[x_t1]
        minParam[self.threadID-1]=minv
		minval[self.threadID-1]=min1	
		print 'Hi!!!'+str(minParam[self.threadID-1])+' '+str(minval[self.threadID-1])#+' '+str(miny)+' '+str(mins)

def print_time(i, threadName):
	#time.sleep(delay)
	#xx=x[i*5:(i+1)*5]+y[i*5:(i+1)*5]
	m=10
	x= np.column_stack((xv[i*m:(i+1)*m].flatten().T,yv[i*m:(i+1)*m].flatten().T,zv[i*m:(i+1)*m].flatten().T,wv[i*m:(i+1)*m].flatten().T))
	print "%s: %s %s" % (threadName, time.ctime(time.time()), x[-1])

#threadLock = threading.Lock()
threads = []
#x=np.asarray([1,2,1,0,1,2,2,3,4,5,4,3,2,1,0])
#y=np.asarray([2,5,4,3,2,1,0,6,7,8,9,8,3,1,3])
#z=np.asarray([0]*3)
# Create new threads
for i in range(noThreads):
	threads.append(myThread(i+1, "Thread-"+str(i+1)))

# Start new Threads
def startAll():
	for i in range(noThreads):
		threads[i].start()

counter=0
#startAll()
minpos=[0]*noThreads
minval=[0.0]*noThreads
minParam=[[]]*noThreads
b = Barrier(noThreads+1, 'b')
b1= Barrier(noThreads+1, 'b1')
b2= Barrier(noThreads+1, 'b2')
b3= Barrier(noThreads+1, 'b3')
b4= Barrier(noThreads+1, 'b4')
check=0
#b=threading.Barrier(11, timeout=5)
# Add threads to thread list

# Wait for all threads to complete

#for t in threads:
#    	t.join()
#b.wait()
breakAll=False
minPrev=10000000
for l in range(1):
	#v1,v2,v3,v4,v5= np.meshgrid(x1,x1,x1,x1,x1[l])
	#breakAll=False
	if(l==0):
		startAll()
	b3.wait('main')
	breakAll=False
	b3.wait2('main')
	counter=0
	for j in range(limit):
		b.wait('main')
		#if(j==0):
			#del v1,v2,v3,v4,v5
		check=1
		try:
			gp.fit(X, y)
		except:
			y[-1]=y[-1]+random.random()/1000.0
			print y[-1] in y[0:-1]	
			gp.fit(X,y)	
		b.wait2('main')
		b1.wait('main')
		b1.wait2('main')
		b2.wait('main')
		b2.wait2('main')
#		check=1;
#		for t in threads:
#			t.run2()
#			print 'Running '+str(t)
#		for t in threads:
#			t.join()
#		b.wait()
		minAll=numpy.argmin(minval)	
		#print str(minval[minAll])+'------'+str(minParam[minAll])
   		#if (abs(f1(minParam[minAll])-minPrev)<thres) and (minPrev<=min(y)):
		if not(minParam[minAll] in X): 
    			X=numpy.row_stack((X,minParam[minAll]))
		else:
			cc=map(lambda a: a+(nprand.random()-0.5)/10.0,minParam[minAll])			
			X=numpy.row_stack((X,cc))
        y = f(X).ravel()
		minPrev=f1(minParam[minAll])
		#global maxprev
		if (y[-1]<prevmax1):
			maxprev=X[-1]
			prevmax1=y[-1]
		print X
		print y
		if (abs(y[-1]-y[-2])<thres) and (abs(y[-2]-y[-3])<thres) and (abs(y[-3]-y[-4])<thres):
			#counter=counter+1
		#if(counter>=2):
			print 'DONE!'+' '+str(f1(minParam[minAll]))+' '+str(minPrev)+' '+str(minParam[minAll])
			breakAll=True
	 		b4.wait('main')
			b4.wait2('main')
			#b.wait2()
            break
		else:
			b4.wait('main')
			b4.wait2('main')
	print '----- Main-OUT -----'+ str(X)+' '+str(y)+' '+str(minParam[minAll])
print '..... Main-OUT .....'

for t in threads:
	t.join()
	
print "Exiting Main Thread"
