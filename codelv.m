function tot_rms = codelv(a,param)
    param=cell2mat(param);
    options = odeset('RelTol', 1e-4, 'NonNegative', [1 2]);
    [t,x] = ode45(@(t,x)lotka_volterra(t,x,param), linspace(0,20,100), [10 10], options);
    paramStd=[1.0,0.05,0.02,0.5];
    [t1,x1] = ode45(@(t,x)lotka_volterra(t,x,paramStd), linspace(0,20,100), [10 10], options);
tot_rms=(sum(sum((x1-x).^2))/length(x))^0.5;


    