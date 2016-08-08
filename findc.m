function close = findc(t)
times=[24.225 30.475 36.725 42.975 49.225 55.475 61.725 67.975];
close=1;
min_d=abs(t-times(close));
for i=1:length(t)
    if(min_d>abs(t-times(close)))
end
    