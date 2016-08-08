function ret = plotode(ti, Yi, lb, ub, x, N, plotyn)
    if plotyn==true
        figure;
        subplot(321)
        plot([lb:ub],x(:,1),'r-',[lb:ub],Yi(ti,1:N),'b-'),xlim([lb,ub])
        subplot(322)
        plot([lb:ub],x(:,2),'r-',[lb:ub],Yi(ti,N+1:2*N),'b-'),xlim([lb,ub])
        subplot(323)
        plot([lb:ub],x(:,3),'r-',[lb:ub],Yi(ti,2*N+1:3*N),'b-'),xlim([lb,ub])
        subplot(324)
        plot([lb:ub],x(:,4),'r-',[lb:ub],Yi(ti,3*N+1:4*N),'b-'),xlim([lb,ub])
        subplot(325)
        plot([lb:ub],x(:,5),'r-',[lb:ub],Yi(ti,4*N+1:5*N),'b-'),xlim([lb,ub])
        subplot(326)
        plot([lb:ub],x(:,6),'r-',[lb:ub],Yi(ti,5*N+1:6*N),'b-'),xlim([lb,ub])
    end
    rmsx=0;
    if(N==58)
        for i=1:6
           rmsx=rmsx+sum((x(:,i)-Yi(ti,(i-1)*N+1:i*N)').^2);
        end
    else
        for i=[1,2,3,4,6]
           rmsx=rmsx+sum((x(:,i)-Yi(ti,(i-1)*N+1:i*N)').^2);
        end
    end
   
    %rms=(rms/(N*6))^0.5;
    ret=rmsx;
