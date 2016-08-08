function tot_rms = code(a,param)
    plotyn=false;
    %param=cell2mat(param);
    %load('param33006.mat')
    %options = odeset('RelTol',1e-4,'AbsTol',1e-4);
    load('x13.mat');
    load('init13.mat');
    init13c=zeros(1,180);
    init13c(1:60)=init13(1:60);
    init13b=zeros(1,174);
    init13b(1:29)=init13(1:29);
    init13b(30:58)=init13(31:59);
    %load('param.mat');
    tot_rms=0;
    [T,Y] = ode45(@(t,g)cycle13(t,g,param),[0.0 10.55 16.0],init13);
    tot_rms=tot_rms+plotode(2,Y,17,46,x13,30,plotyn);
    [T2,Y2] = ode45(@(t,g)mitosis(t,g,param),[16.0 20.0 21.1],Y(3,:));
    %figure;
    plotode(2,Y2,17,46,x13,30,plotyn);
    m=split(Y2(3,:));
    load('x14anew.mat');
    %tot_rms=0;
    %init14a=[m(2:59),m(62:119),m(122:179),m(182:239),m(242:299),m(302:359)];
    init14a=reshape(permute(x14a(1,:,:),[1 3 2]),[58,6]);
    [T3,Y3] = ode45(@(t,g)cycle14a(t,g,param),[24.225 30.475 36.725 42.975 49.225 55.475 61.725 67.975],init14a);
    %tot_rms=0;
    for ii=1:8
        %figure;
        tt=reshape(permute(x14a(ii,:,:),[1 3 2]),[58,6]);
        tot_rms=tot_rms+plotode(ii,Y3,35,92,tt,58,plotyn);
    end
    tot_rms=(tot_rms/(58*7*6+30*5))^0.5;
    %tot_rms=(tot_rms/(180))^0.5;
