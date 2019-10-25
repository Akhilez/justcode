function res = scatter_classes_with_boundary(x, y, pred_bound_x, pred_bound_y, bound_x, bound_y, fig_number)
  p_index = y == 1;
  n_index = y == -1;

  x1p = x(p_index, 1);
  x2p = x(p_index, 2);

  figure(fig_number)
  scatter(x(p_index, 1), x(p_index, 2), 'r')
  hold on
  scatter(x(n_index, 1), x(n_index, 2), 'b')
  plot(pred_bound_x, pred_bound_y, bound_x, bound_y)
  title('Decision Boundaries')
  xlabel('x1')
  ylabel('x2')
  legend('1', '0', 'predicted', 'real')
  saveas(gcf, 'figures/decision.png');

end