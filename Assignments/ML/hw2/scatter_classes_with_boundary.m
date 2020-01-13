function res = scatter_classes_with_boundary(x, y, pred_bound_y, bound_y, fig_number)
  p_index = y == 1;
  n_index = y == -1;

  x1p = x(p_index, 1);
  x2p = x(p_index, 2);

  [boundary_rows, boundary_cols] = size(pred_bound_y);

  figure(fig_number)
  scatter(x(p_index, 1), x(p_index, 2), 'r')
  hold on
  scatter(x(n_index, 1), x(n_index, 2), 'b')
  for i=1:boundary_cols
    x_plot = x(:, 1);
    y_plot = pred_bound_y(:, i);
    plot(x_plot, y_plot)
    hold on
  end
  plot(x(:, 1), bound_y)
  title('Decision Boundaries')
  xlabel('x1')
  ylabel('x2')
  legend('1', '0', '5 epochs', '10 epochs', '50 epochs', '100 epochs', 'real')
  saveas(gcf, 'figures/decision.png');

end