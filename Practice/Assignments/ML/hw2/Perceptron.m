classdef Perceptron < handle
   properties
      weights
   end
   methods
      
      function errors = train_batch(obj, x, y, epochs, lr)
        size_of_x = size(x);
        x = [x, ones(size_of_x(1), 1)];
        errors = zeros(epochs, 1);
        for epoch = 1:epochs
          delta = zeros(1, size_of_x(2)+1);
          error = 0;
          for i = 1:length(y)
            fx = dot(obj.weights, x(i, :));
            err = y(i) - fx;
            delta = delta + lr * err * x(i, :);
            error = error + err * err;
          end
          obj.weights = obj.weights + delta;
          errors(epoch) = error;
        end
        return;
      end
      
      
      function errors = train_incremental(obj, x, y, epochs, lr)
        size_of_x = size(x);
        x = [x, ones(size_of_x(1), 1)];
        errors = zeros(epochs, 1);
        for epoch = 1:epochs
          error = 0;
          for i = 1:length(y)
            fx = dot(obj.weights, x(i, :));
            err = y(i) - fx;
            obj.weights = obj.weights + lr * err * x(i, :);
            error = error + err * err;
          end
          errors(epoch) = error;
        end
        return;
      end


      function errors = train_decaying_lr(obj, x, y, epochs, lr, decay)
        size_of_x = size(x);
        x = [x, ones(size_of_x(1), 1)];
        errors = zeros(epochs, 1);
        for epoch = 1:epochs
          delta = zeros(1, size_of_x(2)+1);
          error = 0;
          for i = 1:length(y)
            fx = dot(obj.weights, x(i, :));
            err = y(i) - fx;
            delta = delta + lr * err * x(i, :);
            error = error + err * err;
          end
          obj.weights = obj.weights + delta;
          errors(epoch) = error;
          lr = lr * decay;
        end
        return;
      end


      function errors = train_adaptive_lr(obj, x, y, epochs, lr, d, D)
        size_of_x = size(x);
        x = [x, ones(size_of_x(1), 1)];
        prev_error = -1;
        errors = zeros(epochs, 1);
        lrs = zeros(epochs, 1);
        epoch = 1;
        while epoch <= epochs
          delta = zeros(1, size_of_x(2)+1);
          error = 0;
          for i = 1:length(y)
            fx = dot(obj.weights, x(i, :));
            err = y(i) - fx;
            delta = delta + lr * err * x(i, :);
            error = error + err * err;
          end
          lrs(epoch) = lr;
          if prev_error == -1
            prev_error = error;
          else
            if error - prev_error > 0
              lr = lr * d;
              continue;
            else
              lr = lr * D;
            end
          end
          obj.weights = obj.weights + delta;
          errors(epoch) = error;
          epoch = epoch +1;
        end
        figure(5)
        plot(1:length(lrs), lrs)
        title('LR vs Epoch')
        xlabel('Epochs')
        ylabel('Learning rate')
        saveas(gcf, 'figures/adaptive_rates.png');

        return;
      end
      
      
      function y_pred = test(obj, x_test)
        size_of_x = size(x_test);
        x_test = [x_test, ones(size_of_x(1), 1)];
        y_pred = zeros(length(x_test), 1);
        for i = 1:length(x_test)
          y = dot(obj.weights, x_test(i, :));
          if y >= 0
            y_pred(i) = 1;
          else
            y_pred(i) = -1;
          end
        end
        return
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
