function y = equation(x)
  %{
  parameter x is a matrix of two columns!
  
  (1)*x1 + (2)*x2 − 2 > 0
  
  %}
  a = 1;
  b = 2;
  c = -2;
  
  y_real = a * x(:,1) + b * x(:, 2) + c;
  
  y = y_real;
  y(y_real>=0) = 1;
  y(y_real<0) = 0;
  
  return
end 