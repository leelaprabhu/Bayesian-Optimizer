function double = split(sing)
    double= zeros(length(sing)*2,1);
    for i = 1:length(sing);
        double(2*i-1)=sing(i);
        double(2*i)=sing(i);
    end