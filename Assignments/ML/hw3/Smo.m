classdef Smo < handle
   properties
      x
      y
      nrows
      ncols
      epsilon
      C
      alpha
      b
      w
   end
   methods

      function obj = Smo(x, y)
        obj.x = x;
        obj.y = y;
        [obj.nrows, obj.ncols] = size(x);
        obj.epsilon = 0.01;
        obj.C = 5;
        obj.alpha = obj.get_init_alpha();
        obj.b = 0;
      end

      function res = train(self)
        itr = 0;
        while itr < 10
          itr = itr + 1;

          % Step 2
          self.w = self.get_init_w();

          % Step 3
          kkts = self.get_kkts();

          % Step 4
          % Step 4a
          [argvalue, argmax] = max(kkts);
          i1 = argmax;

          % Step 4b
          x1 = argvalue;

          % Step 4c
          errors = self.get_errors();
          relative_errors = abs(errors(i1) - errors);
          [argvalue, argmax] = max(relative_errors);

          % Step 4d
          i2 = argmax;

          % Step 4e
          x2 = argvalue;

          % Step 4f
          k11 = self.get_kernel_value(i1, i1);
          k22 = self.get_kernel_value(i2, i2);
          k12 = self.get_kernel_value(i1, i2);
          k = k11 + k22 - 2 * k12;

          alpha1_prev = self.alpha(i1);
          alpha2_prev = self.alpha(i2);

          % Step 5
          alpha2_change = self.y(i2) * relative_errors(i2) / k;
          self.alpha(i2) = self.alpha(i2) + alpha2_change;

          % Step 6
          self.alpha(i1) = self.alpha(i1) + self.y(i1) * self.y(i2) * alpha2_change;

          % Step 7
          self.adjust_alpha_extremes();

          % Step 8
          if self.alpha(i1) < self.C && self.alpha(i1) > 0
            self.b = self.b - errors(i1) - self.y(i1) * (self.alpha(i1) - alpha1_prev) * k11 - self.y(i2) * (self.alpha(i2) - alpha2_prev) * k12;
          elseif self.alpha(i2) < self.C && self.alpha(i2) > 0
            self.b = self.b - errors(i2) - self.y(i1) * (self.alpha(i1) - alpha1_prev) * k12 - self.y(i2) * (self.alpha(i2) - alpha2_prev) * k22;
          else
            b1 = self.b - errors(i1) - self.y(i1) * (self.alpha(i1) - alpha1_prev) * k11 - self.y(i2) * (self.alpha(i2) - alpha2_prev) * k12;
            b2 = self.b - errors(i2) - self.y(i1) * (self.alpha(i1) - alpha1_prev) * k12 - self.y(i2) * (self.alpha(i2) - alpha2_prev) * k22;
            self.b = (b1 + b2) / 2;
          end

          % Step 9
          y_pred = self.classify(self.x);
          classified = all(y_pred == self.y);

          % Step 10
          if classified
            break;
          end

          disp("Iteration: ");
          disp(itr);

        end
      end

      function y_pred = classify(self, x_pred)
        [nrows_x, ~] = size(x_pred);
        y_pred = zeros(1, nrows_x);
        for i = 1:nrows_x
          y_pred_i = dot(self.w, x_pred(i, :)) + self.b;
          if y_pred_i < 0
            y_pred(i) = -1;
          else
            y_pred(i) = 1;
          end
        end
      end

      function none = adjust_alpha_extremes(self)
        for i = 1:self.nrows
          if self.alpha(i) < self.epsilon
            self.alpha(i) = 0;
          elseif self.alpha(i) > self.C
            self.alpha(i) = self.C;
          end
        end
      end

      function errors = get_errors(self)
        errors = zeros(self.nrows, 1);
        for i = 1:self.nrows
          errors(i) = self.get_error_i(i);
        end
      end

      function kkts = get_kkts(self)
        kkts = zeros(self.nrows, 1);
        for i = 1:self.nrows
          kkts_i = self.alpha(i) * (self.y(i) * (dot(self.w, self.x(i, :)) - self.b) - 1);
          kkts(i) = kkts_i;
        end
      end

      function w_sum = get_init_w(obj)
        w_sum = zeros(1, obj.ncols);
        for i = 1:obj.nrows
          w_sum = w_sum + obj.alpha(i) * obj.y(i) * obj.x(i, :);
        end
      end

      function error = get_error_i(obj, i)
        sum_term = 0;
        for j = 1:obj.nrows
          sum_term = sum_term + obj.alpha(j) * obj.y(j) * obj.get_kernel_value(i, j);
        end
        error = sum_term + obj.b - obj.y(i);
      end

      function kernel_out = get_kernel_value(self, i, j)
        kernel_out = dot(self.x(i,:), self.x(j,:));
      end

      function alphas = get_init_alpha(self)
        pos_count = 0;
        neg_count = 0;
        for i = 1:self.nrows
          if self.y(i) < 0
            neg_count = neg_count + 1;
          else
            pos_count = pos_count + 1;
          end
        end
        max_count = max([pos_count, neg_count]);
        min_count = min([pos_count, neg_count]);
        min_alphas = self.get_rand_range(0, self.C, [min_count, 1]);
        remaining_alphas = self.get_rand_range(0, min(min_alphas), [max_count - min_count, 1]);
        reduction = sum(remaining_alphas) / min_count;
        max_alphas = cat(1, min_alphas - reduction, remaining_alphas);
        alphas = zeros(self.nrows, 1);
        if max_count == pos_count
          pos_alphas = max_alphas;
          neg_alphas = min_alphas;
        else
          pos_alphas = min_alphas;
          neg_alphas = max_alphas;
        end
        pos_i = 1;
        neg_i = 1;
        for i = 1:self.nrows
          if self.y(i) < 0
            alphas(i) = neg_alphas(neg_i);
            neg_i = neg_i + 1;
          else
            alphas(i, 1) = pos_alphas(pos_i);
            pos_i = pos_i + 1;
          end
        end
      end

      function rand_vec = get_rand_range(obj, low, high, size)
        rand_vec = (high-low).*rand(size) + low;
      end

   end
end
