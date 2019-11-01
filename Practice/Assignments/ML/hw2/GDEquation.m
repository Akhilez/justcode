classdef GDEquation < handle
  properties
    weights
  end
  methods

    function obj = GDEquation(weights)
      obj.weights = weights;
    end

    function y = equation(obj, x)
      y = obj.weights(1);
      for i = 2:length(obj.weights)
        y = y + obj.weights(i) * (x(i-1) + x(i-1) * x(i-1));
      end
      return;
    end

    function errors = train(obj, x, y, epochs, lr)
      errors = zeros(epochs, 1);
      [nrows, ncols] = size(x);
      for epoch = 1:epochs
        error = 0;
        for i = 1:nrows
          fx = obj.equation(x(i, :));
          err = y(i) - fx;

          obj.weights(1) = obj.weights(1) + lr * err;
          for j = 2:length(obj.weights)
            obj.weights(j) = obj.weights(j) + lr * err * (x(i, j-1) + x(i, j-1) * x(i, j-1));
          end

          error = error + err * err;
        end
        errors(epoch) = error;
      end
    end

    function y_pred = test(obj, x)
      [nrows, ncols] = size(x);
      y_pred = zeros(nrows, 1);
      for i = 1:nrows
        r = obj.equation(x(i,:));
        if r >= 0.1
          r = 1;
        elseif r >= -0.6
          r = 0;
        else
          r = -1;
        end
        y_pred(i) = r;
      end
      return;
    end

    function rate = get_hit_rate(obj, y_pred, y_real)
      rate = 0;
      for i = 1:length(y_pred)
        if y_pred(i) == y_real(i)
          rate = rate + 1;
        end
      end
      rate = rate/length(y_pred);
      return;
    end

  end
end