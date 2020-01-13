function y = equation(x, a, b, c)
  %{
  parameter x is a matrix of two columns!
  
  (a)*x1 + (b)*x2 − c > 0
  
  %}
  
  y_real = a * x(:,1) + b * x(:, 2) + c;
  
  y = y_real;
  y(y_real>=0) = 1;
  y(y_real<0) = -1;
  
  return
end 