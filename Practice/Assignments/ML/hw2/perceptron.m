classdef Perceptron
   properties
      weights
   end
   methods
      
      function errors = train_batch(x, y, epochs, lr)
        errors = zeros(epochs);
        for epoch = 1:epochs
          delta = zeros(length(y));
          error = 0;
          for i = 1:length(y)
            fx = weights * x(i, :);
            err = y(i) - fx;
            delta = delta + lr * err * x(i, :);
            error = error + err * err;
          end
          weights = weights + delta
          errors(epoch) = error;
        end
        return;
      end
      
      
      function errors = train_iterative(x, y, epochs, lr)
        errors = zeros(epochs);
      end
      
      
      function = init_weights(size)
        weights = rand(size, 1);
      end
   end
end