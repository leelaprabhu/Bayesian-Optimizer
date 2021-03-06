function dg2 = cycle14a(t,g2,param)
% dy = zeros(3,1);    % a column vector
% dy(1) = y(2) * y(3);
% dy(2) = -y(1) * y(3);
% dy(3) = -0.51 * y(1) * y(2);
% param= [20.0, 19.608, 16.373, 15.789, 12.185, 11.906, -0.068, -0.073, -0.050, -0.056, -0.038, -0.034, 0.022, 0.019, 0.001, 0.011, -0.166, 0.003, 0.033, -0.014, 0.017, -0.076, -0.015, -0.080, 0.029, -0.018, -0.110, 0.011, -0.001, -0.020, 0.037, -0.027, -0.024, -0.090, 0.045, -0.077, -0.018, -0.106, -0.106, -0.082, -0.137, -0.003, -0.040, 0.050, 0.129, 0.177, 0.097, -0.007, 13.459, -3.500, -3.500, -3.500, -3.500, 8.173, 72.18, 173.87, 31.06, 77.98, 33.60, 0.99, 113.31, 8.99, 14.99, 39.54, 20.75, 77.15];   
R=param(1:6);
T=reshape(param(7:42),[6,6])';
m=param(43:48);
h=param(49:54);
D=param(55:60);
t1_2=param(61:66);
v34=[[24.6800  158.4000    7.2700   61.4500         0   77.1526]
   [26.8400  186.8200    7.4400  103.3100    0.0900    0.4900]
   [11.2300  205.2100    3.9200  141.8400         0    0.3600]
   [12.0600  208.8500    4.1700  186.8200         0    0.5700]
    [8.0800  206.3200    2.8700  203.4200         0    0.5200]
    [7.9700  192.0400    2.8800  195.3900         0    0.8300]
   [13.8300  180.3700    3.3400  211.9200         0    0.4100]
   [18.6900  151.8300    3.7200  213.0500         0    1.6500]];

v93=[[86.2300    6.6400    0.6800   24.0700         0    0.9947]
  [103.4300    8.3200    0.4500    9.4400         0  104.4600]
   [83.6900   16.0600         0    2.4700    0.1300  133.3200]
   [73.2900   28.6800         0    0.0200    0.6100  152.1100]
   [69.7600   24.9100         0    0.5000         0  150.0500]
   [64.8800   23.3200         0         0         0  141.7100]
   [33.4700   17.3000    0.1600         0         0  144.5300]
   [20.0100    8.2100    6.4200         0         0   80.7700]];

bcd=[39.7312500000000;36.9862500000000;35.4050000000000;32.9000000000000;30.7525000000000;29.5362500000000;27.7675000000000;25.7075000000000;24.0962500000000;22.7300000000000;21.4700000000000;19.7575000000000;18.5075000000000;17.2050000000000;15.9187500000000;14.6925000000000;14.0762500000000;12.8012500000000;11.4887500000000;11.0225000000000;9.75125000000000;9.17500000000000;8.59625000000000;8.04125000000000;7.03250000000000;6.93250000000000;6.30500000000000;5.55750000000000;4.86125000000000;4.39875000000000;3.84750000000000;3.91625000000000;3.56000000000000;3.08125000000000;3.08875000000000;2.60250000000000;2.49750000000000;2.39000000000000;2.12500000000000;1.57125000000000;1.58625000000000;1.22000000000000;1.24250000000000;1.75500000000000;1.22875000000000;1.15875000000000;0.800000000000000;0.923750000000000;0.620000000000000;0.598750000000000;0.746250000000000;0.470000000000000;0.562500000000000;0.381250000000000;0.473750000000000;0.726250000000000;0.757500000000000;0.621250000000000];
%v34=[72.18, 173.87, 31.06, 77.98, 33.60, 0.99];
%v93=[113.31, 8.99, 14.99, 39.54, 20.75, 77.15];
%v34=[21.3600  188.1637    7.4250  166.7163    4.2113    0.7275];
%v93=[70.2300   16.9737    2.7525    6.4962    2.6862  123.0125];
times=[24.2250   30.4750   36.7250   42.9750   49.2250   55.4750   61.7250   67.9750];

% R=[20.0, 19.608, 16.373, 15.789, 12.185, 11.906];
% T=[[-0.068, -0.073, -0.050, -0.056, -0.038, -0.034];
%    [0.022, 0.019, 0.001, 0.011, -0.166, 0.003];
%    [0.033, -0.014, 0.017, -0.076, -0.015, -0.080];
%    [0.029, -0.018, -0.110, 0.011, -0.001, -0.020];
%    [0.037, -0.027, -0.024, -0.090, 0.045, -0.077];
%    [-0.018, -0.106, -0.106, -0.082, -0.137, -0.003]];
% m=[-0.040, 0.050, 0.129, 0.177, 0.097, -0.007];
% h=[13.459, -3.500, -3.500, -3.500, -3.500, 8.173];
% D=[0.200, 0.200, 0.200, 0.142, 0.200, 0.200];
% t1_2=[18.000, 7.254, 8.980, 9.577, 12.499, 16.842];
% v34=[72.18, 173.87, 31.06, 77.98, 33.60, 0.99];
% v93=[113.31, 8.99, 14.99, 39.54, 20.75, 77.15];

% R=[     12.812132,   22.786520,  21.588263,  22.815352,  22.037550,  17.954936];
% T=[[    -0.012352,  -0.018824,  -0.017068,  -0.026927,  -0.015665,  -0.014759];
%    [    0.015132,   0.019551,   0.001516,   0.013021,   -0.075834,  0.003674];
%    [    0.026518,   0.003393,   0.013070,   -0.031618,  -0.001788,  -0.064934];
%    [    0.021691,   -0.001633,  -0.052101,  0.016053,   0.004754,   -0.008446];
%    [    0.026775,   -0.068373,  0.001791,   -0.000240,  0.011534,   -0.056754];
%    [    0.034919,   -0.013708,  -0.079439,  -0.023317,  -0.042635,  -0.010401]];
% m= [    -0.032025,  0.032262,   0.025170,   0.052421,   0.021895,   -0.025229];
% h=[     3.646139,   -3.500,     -3.500,     -3.500,     -3.500,     -3.753219];
% D=[     0.00,       0.00,       0.00,       0.00,       0.00,       0.00];
% t1_2=[  19.838265,  5.990225,   7.335732,   6.147511,   7.866904,   13.204312];
% v34=[   72.18,      173.87,     31.06,      77.98,      33.60,      0.99];
% v93=[   113.31,     8.99,       14.99,      39.54,      20.75,      77.15];

% bcd=[65.4100
%    62.1200
%    59.4300
%    57.8200
%    55.3500
%    53.1100
%    50.8200
%    49.5500
%    47.1200
%    45.5100
%    43.5400
%    42.3100
%    40.5700
%    38.9700
%    37.8100
%    36.3900
%    34.9600
%    33.6400
%    32.6700
%    31.4700
%    30.5200
%    29.6800
%    28.6400
%    27.8300
%    27.0400
%    25.9100
%    25.3200
%    24.7800
%    23.7800
%    23.1400
%    22.2900
%    21.8500
%    21.2000
%    20.5800
%    20.1300
%    19.6000
%    19.2300
%    18.5600
%    18.0500
%    17.6900
%    17.3000
%    16.6200
%    16.1800
%    15.6300
%    15.2300
%    14.8400
%    14.3200
%    13.9200
%    13.5900
%    13.3800
%    13.2400
%    13.1500
%    13.0500
%    12.8500
%    12.7600
%    12.7900
%    12.3800
%    12.1000]';
% bcd=[61.2150
%    58.8762
%    57.5663
%    55.3487
%    53.6500
%    52.5938
%    51.3050
%    49.2875
%    47.9163
%    46.6938
%    45.6725
%    44.2463
%    43.1363
%    41.9650
%    40.8588
%    39.7288
%    38.3625
%    38.0400
%    36.8912
%    36.4438
%    35.2188
%    34.6212
%    34.0575
%    33.4763
%    32.5712
%    32.2788
%    31.5300
%    30.8425
%    30.1463
%    29.5750
%    28.9100
%    28.8925
%    28.3038
%    27.7575
%    27.5087
%    26.9387
%    26.6250
%    26.3438
%    25.9138
%    25.0437
%    24.8100
%    24.2325
%    24.0287
%    24.1650
%    23.3887
%    23.0925
%    22.4250
%    22.2475
%    21.6538
%    21.3287
%    21.1325
%    20.5425
%    20.3012
%    19.6775
%    19.3725
%    19.0063
%    18.0075
%    17.5900];
% bcd=[41.5500
%    38.8425
%    37.2938
%    34.8300
%    32.7075
%    31.5200
%    29.7812
%    27.7487
%    26.1662
%    24.8163
%    23.5837
%    21.8887
%    20.6575
%    19.3750
%    18.0975
%    16.8850
%    16.2812
%    15.0163
%    13.7112
%    13.2550
%    11.9825
%    11.4088
%    10.8350
%    10.2787
%     9.2725
%     9.1650
%     8.5275
%     7.7737
%     7.0725
%     6.6075
%     6.0375
%     6.0900
%     5.7212
%     5.2300
%     5.2175
%     4.7113
%     4.5862
%     4.4625
%     4.1738
%     3.5938
%     3.5863
%     3.1912
%     3.1825
%     3.6725
%     3.1162
%     3.0125
%     2.5900
%     2.6637
%     2.3187
%     2.2713
%     2.4013
%     2.1075
%     2.1637
%     1.9387
%     1.9850
%     2.1913
%     2.1788
%     1.9938];
%dg=zeros(348,1);
% for a=1:6
%     for i=1:58
%         Tv=T(a,1)*g(i)+T(a,2)*g(58+i)+T(a,3)*g(58*2+i)+T(a,4)*g(58*3+i)+T(a,5)*g(58*4+i)+T(a,6)*g(58*5+i);
%         ua=Tv+m(a)*bcd(i)+h(a);
%         if(i==1)
%            vv=(v34(a)-g((a-1)*58+i))+(g((a-1)*58+i+1)-g((a-1)*58+i));
%         elseif (i==58)
%            vv=(g((a-1)*58+i-1)-g((a-1)*58+i))+(v93(a)-g((a-1)*58+i));
%         else
%            vv=(g((a-1)*58+i-1)-g((a-1)*58+i))+(g((a-1)*58+i+1)-g((a-1)*58+i));    
%         end
%         lmbd=log(2.0)/t1_2(a);
%         dg((a-1)*58+i)=R(a)*0.5*(ua/((ua^2.0+1)^0.5)+1)+D(a)*vv-lmbd*g((a-1)*58+i);
%     end 
% end    

dg=zeros(6,58);
g=reshape(g2,[58,6])';
[aa,tp]=min(abs(t-times));
for a=1:6
    for i=1:58
        Tv=T(a,1)*g(1,i)+T(a,2)*g(2,i)+T(a,3)*g(3,i)+T(a,4)*g(4,i)+T(a,5)*g(5,i)+T(a,6)*g(6,i);
        ua=Tv+m(a)*bcd(i)+h(a);
        if(i==1)
           vv=(v34(tp,a)-g(a,i))+(g(a,i+1)-g(a,i));
        elseif (i==58)
           vv=(g(a,i-1)-g(a,i))+(v93(tp,a)-g(a,i));
        else
           vv=(g(a,i-1)-g(a,i))+(g(a,i+1)-g(a,i));    
        end
        lmbd=log(2.0)/t1_2(a);
        dg(a,i)=R(a)*0.5*(ua/((ua^2.0+1)^0.5)+1)+D(a)*vv-lmbd*g(a,i);
    end 
end   
%size(dg)
dg2=reshape(dg',[1,348])';

% dg=[0.0]*58*6
%     for a in range(6): #gene
%         for i in range(58): #nucleus
%             Tv=T[a][0]*g[i]+T[a][1]*g[58+i]+T[a][2]*g[58*2+i]+T[a][3]*g[58*3+i]+T[a][4]*g[58*4+i]+T[a][5]*g[58*5+i]
%             ua=Tv+m[a]*bcd[i]+h[a]T
%             if(i==0):
%                 vv=(v34[a]-g[a*58+i])+(g[a*58+i+1]-g[a*58+i])
%             elif(i==57):
%                 vv=(g[a*58+i-1]-g[a*58+i])+(v93[a]-g[a*58+i])
%             else:
%                 vv=(g[a*58+i-1]-g[a*58+i])+(g[a*58+i+1]-g[a*58+i])
%             lmbd=2.0/math.exp(t1_2[a])
%             dg[a*58+i]=R[a]+0.5*(ua/((ua**2.0+1)**0.5)+1)+D[a]*vv-lmbd
%     return dg
    
