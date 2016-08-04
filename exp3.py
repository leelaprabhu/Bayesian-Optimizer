import random
arms=[]
for i in range(1,5):
    for j in range(i+1,6):
        arms.append([i,j])

randgen=[]
for i in range(1000):
    randgen.append(random.sample(range(1,6), 2))

wh=[1]*len(arms)
ph=[0]*len(arms)
ws=[1]*len(arms)
ps=[0]*len(arms)

gamma=0.1
def findgreater(roulette,num):
    for i in range(0,len(roulette)):
        if roulette[i]>num:
            return i+1

def reward(randarm,sel_arm):
    rew=0
    if((sel_arm) in randarm):
        rew=1
    return rew


for t in range(len(randgen)):
        for i in range(0,len(arms)):
            ps[i]=(1-gamma)*ws[i]/sum(ws)+gamma/len(arms)
        for i in range(0,len(arms)):
            ph[i]=(1-gamma)*wh[i]/sum(wh)+gamma/len(arms)
        roulettes=[]
        for i in range(0,len(arms)):
            roulettes.append(sum(ps[0:i+1]))
        rouletteh=[]
        for i in range(0,len(arms)):
            rouletteh.append(sum(ph[0:i+1]))
        sel_arm1s=findgreater(roulettes,random.random())
        sel_arm2s=findgreater(roulettes,random.random())
        while(sel_arm2s==sel_arm1s):
            sel_arm2s=findgreater(roulettes,random.random())
        sel_arm1h=findgreater(rouletteh,random.random())
        sel_arm2h=findgreater(rouletteh,random.random())
        while(sel_arm2h==sel_arm1h):
            sel_arm2h=findgreater(rouletteh,random.random())
        #randarm=randgen[t]#random.sample(range(1,6), 2)
        x_t_1s= reward([sel_arm1h,sel_arm2h], sel_arm1s)
        x_t_2s= reward([sel_arm1h,sel_arm2h], sel_arm2s)
        x_t_1h= 2- reward([sel_arm1s,sel_arm2s], sel_arm1h)
        x_t_2h= 2- reward([sel_arm1s,sel_arm2s], sel_arm2h)
        j1s=arms.index(sel_arm1s)
        ws[j1s]=ws[j1s]*math.exp(gamma/len(arms)*x_t_1s/ps[j1s])
        j2s=arms.index(sel_arm2s)
        ws[j2s]=ws[j2s]*math.exp(gamma/len(arms)*x_t_2s/ps[j2s])
        j1h=arms.index(sel_arm1h)
        wh[j1h]=wh[j1h]*math.exp(gamma/len(arms)*x_t_1h/ph[j1h])
        j2h=arms.index(sel_arm2h)
        wh[j2h]=ws[j2h]*math.exp(gamma/len(arms)*x_t_2h/p[j2h])


randgen=[[2,1]]*100+[[2,3]]*100+[[4,5]]*100+[[3,4]]*100+[[2,4]]*100
random.shuffle(randgen)

sum(x.count(1) for x in randgen)
sum(x.count(2) for x in randgen)
sum(x.count(3) for x in randgen)
sum(x.count(4) for x in randgen)
sum(x.count(5) for x in randgen)

