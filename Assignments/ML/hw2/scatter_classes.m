function res = scatter_classes(x, y)
  p_index = y == 1;
  n_index = y == -1;
  
  x1p = x(p_index, 1);
  x2p = x(p_index, 2);
  
  scatter(x(p_index, 1), x(p_index, 2), 'r');
  hold on
  scatter(x(n_index, 1), x(n_index, 2), 'b');

end