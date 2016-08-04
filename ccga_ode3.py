import math
import numpy
import random
import matlab.engine
eng = matlab.engine.start_matlab()
import numpy as np
import numpy.random as nprand
from sklearn import gaussian_process
from numpy import genfromtxt
from scipy.stats import norm

def initPop(noIndi, prec):
    pop= numpy.random.random_integers(0, int(math.pow(2,prec)-1),noIndi)
    return pop

#def rastrigin1(x):
#global countFunc
#countFunc=countFunc+1
#f_x=3.0*len(x)
#for i in range(0,len(x)):
#    f_x=f_x + (x[i]*x[i] - 3.0*math.cos(2.0*math.pi*x[i]))
#return f_x

def rastriginx(x):
    global gp
    y_pred, sigma2_pred = gp.predict(np.asarray(x).reshape(1,-1), eval_MSE=True)
    return list(y_pred/(sigma2_pred**0.5))[0]

def rastriginy(x):
    global gp
    y_pred, sigma2_pred = gp.predict(np.asarray(x).reshape(1,-1), eval_MSE=True)
    return list(y_pred/(sigma2_pred**0.5))[0]

def rastriginz(x):
    global gp
    global p
    global w
    for i in range(0,len(arms)):
        p[i]=(1-gamma)*w[i]/sum(w)+gamma/len(arms)
    roulette1=[]
    for i in range(0,len(arms)):
        roulette1.append(sum(p[0:i+1]))
    sel_arm=findgreater1(roulette1,random.random())
    return [arms(sel_arm,x),arm]

def rastrigin(param):
    #print "----------------------------rastrigin----------------------------------"
    global gp
    global sel_arm
    global iter
    x=sel_arm
    y_pred, sigma2_pred = gp.predict(np.asarray(param).reshape(1,-1), eval_MSE=True)
    if (x==0):
        return list(y_pred-2.0*(100.0/iter)*(sigma2_pred**0.5))[0]
    elif (x==1):
        return list(y_pred-2.0*(100.0/iter)*(sigma2_pred**0.5))[0]
    elif (x==2):
        return list(y_pred-2.0*(100.0/iter)*(sigma2_pred**0.5))[0]
    elif (x==3):
        #return list(expImp(y_pred, sigma2_pred))[0]
        return list(y_pred-2.0*(100.0/iter)*(sigma2_pred**0.5))[0]
    else:
        return list(y_pred-2.0*(100.0/iter)*(sigma2_pred**0.5))[0]

#def rastrigin(param):
#    param=list(param)
#    param=map(lambda x: float(x), param)
#    return eng.code(1,param)

def f1(param):
    param=list(param)
    param=map(lambda x: float(x), param)
    return eng.code2(1,param)

def process2(x, lower, upper, prec):
    ret=[]
    step=(upper-lower)/(math.pow(2,prec)-1)
    for i in range(0, len(x)):
        ret.append(lower+step*x[i])
    return ret

def process(x, lower, upper, prec):
    #print "----------------------------process----------------------------------"
    ret=[]
    upper=np.asarray([0.0]*66)
    lower=np.asarray([0.0]*66)
    upper[0:6]=30.0
    lower[0:6]=10.0
    upper[6:48]=0.3
    lower[6:48]=-0.3
    upper[48]=30.0
    lower[48]=0.0
    upper[49:53]=-3.5
    lower[49:53]=-3.5
    upper[53]=30.0
    lower[53]=-30.0
    upper[54:60]=0.3
    lower[54:60]=0.0
    upper[60:66]=20.0
    lower[60:66]=5.0
    step=(upper-lower)/(math.pow(2,prec)-1)
    ret=lower+step*x
    return ret

def findFit(i,indi,bestIndi,lower,upper,prec):
    #print "----------------------------findFit----------------------------------"
    fit2=[]
    for j in range(0,len(indi)):
        temp=bestIndi[:]
        temp[i]=indi[j]
        fit2.append(rastrigin(process(temp,lower,upper,prec)))
    fitMax= max(fit2)*1.05
    fit3=[]
    for j in range(0, len(indi)):
        fit3.append(fitMax-fit2[j])
    return fit3

def scaleFit(fits,fsub):
    #print "----------------------------scaleFit----------------------------------"
    fits2=[]
    for i in range(0, len(fits)):
        fits2.append([])
        for j in range(0, len(fits[i])):
            fits2[i].append(fits[i][j]-fsub)
    #print "~~~~~~~~~~~~~~~~~~~"
    #print fits2
    #print "~~~~~~~~~~~~~~~~~~~"
    #print fits
    #print "~~~~~~~~~~~~~~~~~~~"
    #print fsub
    #print "~~~~~~~~~~~~~~~~~~~"
    return fits2

def findBest(noVar, indi, fit):
    bestIndi=[]
    for i in range(0,noVar):
        bestIndi.append(indi[i][fit[i].index(max(fit[i]))])
    return bestIndi

def format(parent):
    #print "----------------------------format----------------------------------"
    str_parent=''
    for i in range(0,len(parent)):
        str_parent= str_parent+ '{0:016b}'.format(parent[i])
    return str_parent

def mutate(sample):
    #print "----------------------------mutate----------------------------------"
    mutant=''
    for i in range(len(sample)):
        prob= random.random()
        if (prob<(1.0/len(sample))):
            if (sample[i]=='1'):
                mutant=mutant+'0'
            else:
                mutant=mutant+'1'
        else:
            mutant=mutant+sample[i]
    return mutant

def crossover(parent1,parent2):
    #print "----------------------------Xover----------------------------------"
    chld=''
    p1=random.randint(0,len(parent1)-2)
    p2=p1
    while(p2==p1):
        p2=random.randint(0,len(parent2)-2)
    pt1=min(p1,p2)
    pt2=max(p1,p2)
    ch=random.randint(0,1)
    if ch==0:
        chld=parent1[0:pt1+1]+parent2[pt1+1:pt2+1]+parent1[pt2+1:len(parent2)]
    else:
        chld=parent2[0:pt1+1]+parent1[pt1+1:pt2+1]+parent2[pt2+1:len(parent2)]
    return chld

def findgreater(roulette,num):
    #print "----------------------------findGrt----------------------------------"
    for i in range(0,len(roulette)):
        if roulette[i]>num:
            return i

def findgreater1(roulette1,num1):
    #print "----------------------------findGrt1----------------------------------"
    for j in range(0,len(roulette1)):
        if roulette1[j]>num1:
            return j

def select2(indis,fitness_scaled):
    #print "----------------------------sel2----------------------------------"
    #print indis
    #print fitness_scaled
    roulette=[]
    for i in range(0,len(indis)):
        roulette.append(sum(fitness_scaled[0:i+1]))
    num1= findgreater(roulette,random.random()*roulette[len(roulette)-1])
    num2=num1
    while(num1==num2):
        #print "----------------------------HERE!!!!!!!----------------------------------"
        #print fitness_scaled[0:i+1]
        num2= findgreater(roulette,random.random()*roulette[len(roulette)-1])
    parent1= indis[num1]
    parent2= indis[num2]
    return ['{0:016b}'.format(parent1),'{0:016b}'.format(parent2)]

def selectPop(pop,fit):
    #print "----------------------------selPop----------------------------------"
    parents=[[]]
    for i in range(0,len(pop)):
        parents.append(select2(pop,fit))
    xxx=parents.pop(0)
    return parents

def genOperate(pop, noVar):
    #print "----------------------------genOp----------------------------------"
    popNew=[]
    for i in range((len(pop)-1)):
        parent1= pop[i][0]
        parent2= pop[i][1]
        xOverProb=random.random()
        if(xOverProb>0.4):
            chld=crossover(parent1,parent2)
        else:
            if(random.random()>0.5):
                chld=parent1
            else:
                chld=parent2
        chld2=mutate(chld)
        popNew.append(int(chld2,2))
    return popNew

#countFunc=0
#prec=16
#noVar=66
#upper=5.12
#lower=-5.12
#noIndi=20
#windowSize=5
#fprev=[0.0]*windowSize

#gen=0
#pop=[]
#fit=[]
#results=[]
#counts=[]

dim=66
noPoint=2
X=np.concatenate((nprand.uniform(10.0,30.0,[noPoint,6]),nprand.uniform(-0.3,0.3,[noPoint,42]),nprand.uniform(0.0,30.0,[noPoint,1]),[[-3.5]*4]*noPoint,nprand.uniform(-30.0,30.0,[noPoint,1]),nprand.uniform(0.0,0.3,[noPoint,6]),nprand.uniform(5.0,20.0,[noPoint,6])),axis=1)
#param1=[10.0613413, 12.3620966, 11.2616159, 15.0797284, 12.1179522, 18.767834, 0.162769512, -0.294598306, 0.18776379, -0.281670863, -0.239565118, -0.094149691, 0.281844816, -0.0940764477, -0.187223621, 0.125543603, -0.214479286, -0.289269856, 0.132446784, -0.00166170748, -0.0772945754, -0.289269856, -0.102838178, -0.229457542, -0.16824445, -0.00101167315, 0.0690821698, 0.0514763104, 0.0855802243, 0.194676127, -0.168876173, -0.0479606317, 0.144953079, -0.261730373, 0.26311284, 0.247621881, 0.208592355, 0.226390478, -0.0701075761, -0.1835523, 0.00485694667, -0.024010071, 0.112579538, 0.264495308, 0.207008469, 0.229704738, -0.298919661, -0.253820096, 25.4428931, -3.5, -3.5, -3.5, -3.5, -25.1338979, 0.132121767, 0.243891051, 0.18852369, 0.230583658, 0.0625360494, 0.032011902, 5.94346532, 7.96932937, 16.2998398, 5.23460746, 7.86221103, 12.8130007]
param=[2.98825055e+01, 1.06863508e+01, 1.06668193e+01, 1.06085298e+01, 2.16572824e+01, 1.02856489e+01, -1.62320897e-01, -1.14520485e-01, -1.80956741e-02, -2.57862211e-02, -9.47585260e-04, -1.59931334e-01, -7.59304189e-02, 2.96475166e-01, -1.66514076e-01, -1.81872282e-02, 1.35825132e-01, 2.99313344e-01, 2.18040742e-01, -1.23840696e-01, -1.85717556e-02, -6.56397345e-02, -2.80050355e-01, -2.89901579e-01, -7.75875486e-02, -7.23735409e-03, -1.29855802e-01, -7.32890822e-03, 1.95170520e-01, -2.26024262e-01, 1.12158389e-01, -2.40727855e-01, -1.66170748e-03, -4.13366903e-03, -9.87548638e-02, -1.52964065e-01, -1.59583429e-01, 7.67681392e-03, -6.46143282e-02, -1.13128862e-01, 2.15523003e-01, 2.99990845e-01, 4.81620508e-02, 2.80068666e-01, 2.20238041e-01, 2.99990845e-01, -1.45236896e-01, -2.66619364e-01, 1.49823758e+01, -3.50000000e+00, -3.50000000e+00, -3.50000000e+00, -3.50000000e+00, -2.76447700e+00, 2.99620050e-01, 2.99995422e-01, 2.99995422e-01, 2.99995422e-01, 1.85882353e-01, 2.99995422e-01, 1.99997711e+01, 1.23650721e+01, 1.89599451e+01, 1.97534905e+01, 1.23092241e+01, 1.48352026e+01]
maxprev=X[0]
#param = genfromtxt('param.csv', delimiter=',')
#X=numpy.row_stack((X,param1))
#X=numpy.row_stack((X,param))
#y = np.asarray([f1(X[0]),f1(X[1])])
y=map(lambda x: f1(x), X)
gp = gaussian_process.GaussianProcess(corr='linear', theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
gp.fit(X, y)
best=f1(param)

def findMin():
    #print "----------------------------findMin----------------------------------"
    countFunc=0
    prec=16
    noVar=66
    upper=5.12
    lower=-5.12
    noIndi=20
    windowSize=5
    fprev=[0.0]*windowSize
    gen=0
    pop=[]
    fit=[]
    results=[]
    counts=[]
    for i in range(0,noVar):
        pop.append(initPop(noIndi, prec))
    for i in range(0,noVar):
        pop.append(initPop(noIndi, prec))
    bestIndi= findBest(noVar,pop,[[0.0]*noIndi]*noVar)
    for i in range(0,noVar):
        fit.append(findFit(i, pop[i], bestIndi, lower, upper, prec))
    while(200):
        gen= gen+1
        fmin=min(min(fit))
        xxx=fprev.pop(0)
        fprev.append(fmin)
        for i in range(0,noVar):
            results.append(rastrigin(process(bestIndi,lower,upper,prec)))
            print str(results[-1])
            fit2=scaleFit(fit,min(fprev))
            if((min(fit2[i])==0.0)and(max(fit2[i])==0.0)):
                cd=map(lambda ab: ab+nprand.random()/100000.0,fit2[i])
            else:
                cd=fit2[i]
            pop2=selectPop(pop[i],cd)
            pop3=genOperate(pop2, noVar)
            pop3.append(pop[0][fit2[0].index(max(fit2[0]))])
            fit3= findFit(i, pop3, bestIndi, lower, upper, prec)
            pop[i]=pop3
            fit[i]=fit3
            bestIndi= findBest(noVar,pop,fit)
        if(len(results)>=10):
            if(abs(results[-1]-results[-2])<=0.001):
                break
    return process(bestIndi,lower,upper,prec)

def expImp(y_pred, sigma2_pred):
    global X
    global y
    tau=X[np.argmax(y)]
    ei= (y_pred-tau)*norm.cdf((y_pred-tau)/(sigma2_pred**0.5))-(sigma2_pred**0.5)*norm.cdf((y_pred-tau)/sigma2_pred)
    return ei

arms = range(5)
w = [1]*len(arms)
p = [0]*len(arms)
gamma=0.7
sel_arm=0

for i in range(5000000):
    iter = i+1.0
    #print "----------------------------MainLoop----------------------------------"
    for j in range(0,len(arms)):
        p[j]=(1-gamma)*w[j]/sum(w)+gamma/len(arms)
    roulette1=[]
    for j in range(0,len(arms)):
        roulette1.append(sum(p[0:j+1]))
    print ">>>>>>>>>> "+str(roulette1)+" <<<<<<<<<<"
    sel_arm=findgreater1(roulette1,random.random())
    cc=findMin()
    if (((sum((np.asarray(X[-1])-np.asarray(cc))**2.0)**0.5)<=40.0)and(i<10)):
        cc=np.concatenate((nprand.uniform(10.0,30.0,[1,6]),nprand.uniform(-0.3,0.3,[1,42]),nprand.uniform(0.0,30.0,[1,1]),[[-3.5]*4]*1,nprand.uniform(-30.0,30.0,[1,1]),nprand.uniform(0.0,0.3,[1,6]),nprand.uniform(5.0,20.0,[1,6])),axis=1)
    X=numpy.row_stack((X,cc))
    y_pred1, sigma2_pred1 = gp.predict(np.asarray(param).reshape(1,-1), eval_MSE=True)
    y_pred2, sigma2_pred2 = gp.predict(np.asarray(X[-1]).reshape(1,-1), eval_MSE=True)
    print X
    print "-----------"+str(rastrigin(param))+" "+str(rastrigin(X[-1]))+" "+str(y_pred1)+" "+str(sigma2_pred1**0.5)+" "+str(y_pred2)+" "+str(sigma2_pred2**0.5)+" "+str(best)+"-----------"
    y=np.concatenate((y,[f1(X[-1])]))
    x_t=1.0-(y[-1]/500.0)
    print "**************** "+str(sel_arm)+" "+str(x_t)+" ****************"
    w[sel_arm]=w[sel_arm]*math.exp(gamma/len(arms)*x_t/p[sel_arm])
    print y
    gp.fit(X, y)