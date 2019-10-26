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
      size_of_x = size(x);
      nrows = size_of_x(1);
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

  end
end