import math
import numpy
import random
import matlab.engine
eng = matlab.engine.start_matlab()
import numpy as np

def initPop(noIndi, prec):
    pop= numpy.random.random_integers(0, int(math.pow(2,prec)-1),noIndi)
    return pop

def rastrigin1(x):
    #global countFunc
    #countFunc=countFunc+1
    f_x=3.0*len(x)
    for i in range(0,len(x)):
        f_x=f_x + (x[i]*x[i] - 3.0*math.cos(2.0*math.pi*x[i]))
    return f_x

def rastriginyy(x):
    global gp
    y_pred, sigma2_pred = gp.predict(np.asarray(x).reshape(1,-1), eval_MSE=True)
    return list(y_pred-(sigma2_pred**0.5))[0]

def rastrigin(param):
    param=list(param)
    param=map(lambda x: float(x), param)
    return eng.code2(1,param)

def f1(param):
    param=list(param)
    param=map(lambda x: float(x), param)
    return eng.code2(1,param)

def process(x, lower, upper, prec):
    ret=[]
    step=(upper-lower)/(math.pow(2,prec)-1)
    for i in range(0, len(x)):
        ret.append(lower+step*x[i])
    return ret

def process(x, lower, upper, prec):
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
    fit2=[]
    global countFunc
    for j in range(0,len(indi)):
        temp=bestIndi[:]
        temp[i]=indi[j]
        countFunc= countFunc+1
        fit2.append(rastrigin(process(temp,lower,upper,prec)))
    fitMax= max(fit2)*1.05
    fit3=[]
    for j in range(0, len(indi)):
        fit3.append(1/fit2[j])
    return fit3

def scaleFit(fits,fsub):
    fits2=[]
    for i in range(0, len(fits)):
        fits2.append([])
        for j in range(0, len(fits[i])):
            fits2[i].append(fits[i][j]-fsub)
    return fits2

def findBest(noVar, indi, fit):
    bestIndi=[]
    for i in range(0,noVar):
        bestIndi.append(indi[i][fit[i].index(max(fit[i]))])
    return bestIndi

def format(parent):
    str_parent=''
    for i in range(0,len(parent)):
        str_parent= str_parent+ '{0:016b}'.format(parent[i])
    return str_parent

def mutate(sample):
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
    for i in range(0,len(roulette)):
        if roulette[i]>num:
            return i

def select2(indis,fitness_scaled):
    roulette=[]
    for i in range(0,len(indis)):
        roulette.append(sum(fitness_scaled[0:i+1]))
    num1= findgreater(roulette,random.random()*roulette[len(roulette)-1])
    num2=num1
    while(num1==num2):
        num2= findgreater(roulette,random.random()*roulette[len(roulette)-1])
    parent1= indis[num1]
    parent2= indis[num2]
    return ['{0:016b}'.format(parent1),'{0:016b}'.format(parent2)]

def selectPop(pop,fit):
    parents=[[]]
    for i in range(0,len(pop)):
        parents.append(select2(pop,fit))
    xxx=parents.pop(0)
    return parents

def genOperate(pop, noVar):
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

minAll=100.0
minRes=[0.0]*66
for i in range(0,noVar):
    pop.append(initPop(noIndi, prec))

bestIndi= findBest(noVar,pop,[[0.0]*noIndi]*noVar)

for i in range(0,noVar):
    fit.append(findFit(i, pop[i], bestIndi, lower, upper, prec))

while(1):
    gen= gen+1
    fmin=min(min(fit))
    xxx=fprev.pop(0)
    fprev.append(fmin)
    for i in range(0,noVar):
        results.append(rastrigin(process(bestIndi,lower,upper,prec)))
        if(minAll>results[-1]):
            minAll=results[-1]
            minRes=bestIndi
        print str(results[-1])+" "+str(minAll)
        print process(minRes,lower,upper,prec)
        fit2=scaleFit(fit,min(fprev))
        pop2=selectPop(pop[i],fit2[i])
        pop3=genOperate(pop2, noVar)
        pop3.append(pop[0][fit2[0].index(max(fit2[0]))])
        fit3= findFit(i, pop3, bestIndi, lower, upper, prec)
        pop[i]=pop3
        fit[i]=fit3
        bestIndi= findBest(noVar,pop,fit)

#def rastrigin(x):
#    f_x=3.0*len(x)
#    for i in range(0,len(x)):
#        f_x=f_x + (x[i]*x[i] - 3.0*math.cos(2.0*math.pi*x[i]))
#    return f_x

#fit=[]
#for kk in range(s):
#    fit.append([])
#    for j in range(noIndi):
#        temp=[]
#        for k in range(noVar):
#            temp.append(pop[k][j])
#        fitx=rastrigin(process(temp,lower,upper,prec))
#        fit[kk].append(500-fitx)

#bestIndi=[]
#for i in range(0,noVar):
#    bestIndi.append(pop[i][fit.index(max(fit))])

