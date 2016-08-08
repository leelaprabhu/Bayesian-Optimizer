function dg2 = mitosis(t,g2,param)

D=param(55:60);
t1_2=param(61:66);

% v16=[101.62,    23.0600,    24.7100,    50.1900,    0.0,    16.3800];
% v47=[115.92,    1.94,       12.8200,    32.8900,    0.0,    3.0800];

v16=[78.77,    104.27,    28.35,    43.22,    0.0,    20.69];
v47=[104.26,    7.56,       15.42,    40.34,    0.0,    26.88];

% dg=zeros(60,1);
% for a=1:6
%     for i=1:30
%         if(i==1)
%            vv=(v16(a)-g((a-1)*30+i))+(g((a-1)*30+i+1)-g((a-1)*30+i));
%         elseif (i==30)
%            vv=(g((a-1)*30+i-1)-g((a-1)*30+i))+(v47(a)-g((a-1)*30+i));
%         else
%            vv=(g((a-1)*30+i-1)-g((a-1)*30+i))+(g((a-1)*30+i+1)-g((a-1)*30+i));    
%         end
%         lmbd=log(2.0)/t1_2(a);
%         dg((a-1)*30+i)=D(a)*vv-lmbd*g((a-1)*30+i);
%     end 
% end    

dg=zeros(6,30);
g=reshape(g2,[30,6])';
for a=1:6
    for i=1:30
        if(i==1)
           vv=(v16(a)-g(a,i))+(g(a,i+1)-g(a,i));
        elseif (i==30)
           vv=(g(a,i-1)-g(a,i))+(v47(a)-g(a,i));
        else
           vv=(g(a,i-1)-g(a,i))+(g(a,i+1)-g(a,i));    
        end
        lmbd=log(2.0)/t1_2(a);
        dg(a,i)=D(a)*vv-lmbd*g(a,i);
    end 
end   
dg2=reshape(dg',[1,180])';


    
