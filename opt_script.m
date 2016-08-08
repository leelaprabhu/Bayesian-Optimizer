load('param33006.mat')
c=linspace(10.0,30.0,1000);
%t-5,20... D 0.0,0.3
res=zeros(1,100);
%for j=1:100
for k=1:1000
    %param(11)=c(j);
    param(5)=c(k);
    res(k)=code(param,false);
end
%end
min(res)