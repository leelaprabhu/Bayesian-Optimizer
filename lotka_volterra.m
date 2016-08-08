function dx = lotka_volterra(t, x, param)
  dx = [0; 0];
  alpha = param(1); 
  beta = param(2); 
  delta = param(3);
  gamma = param(4);

  dx(1) = alpha * x(1) - beta * x(1) * x(2);
  dx(2) = delta * x(1) * x(2) - gamma * x(2);