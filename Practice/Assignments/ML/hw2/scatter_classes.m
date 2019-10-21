function res = scatter_classes(x, y)
  disp('hi');
  
  p_index = y == 1;
  n_index = y == 0;
  
  disp('p_indices');
  disp(p_index);
  
  x1p = x(p_index, 1);
  x2p = x(p_index, 2);
  
  disp('x1p');
  disp(x1p);
  disp('x2p');
  disp(x2p);
  
  scatter(x(p_index, 1), x(p_index, 2), 'r');
  hold on
  scatter(x(n_index, 1), x(n_index, 2), 'b');

end