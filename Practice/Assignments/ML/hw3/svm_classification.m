% ------------ Preprocessing ------------------

data = textread('data.txt');
[nrows, ncols] = size(data);

x = data(:, 1:end-1);
y = data(:, end);

% Scaling X from 0 to 1 for each column
for i = 1:ncols-1
  V = x(:,i);
  maxV = max(V(:));
  minV = min(V(:));
  x(:,i) = (V - minV) / (maxV - minV);
end

% Making y = 0 to y = -1
y(y == 0) = -1;

% Splitting into training and testing sets
split_index = floor(nrows * 0.8);

x_train = x(1:split_index, :);
y_train = y(1:split_index, :);
x_test = x(split_index:end, :);
y_test = y(split_index:end, :);

% ------------------- SVM ----------------------------

svm = Smo(x_train, y_train);
svm.train();
y_preds = svm.classify(x_test).';

% ------------------- Analysing results -------------------

% disp(y_test);
% disp(y_preds);

% Confusion Matrix
cmat = confusionmat(y_test, y_preds);
disp("Confusion Matrix: ");
disp(cmat);

% Accuracy
rate = 0;
for i = 1:length(y_preds)
  if y_preds(i) == y_test(i)
    rate = rate + 1;
  end
end
rate = rate/length(y_preds);
disp("Accuracy = ");
disp(rate);

% --------------------------PLOTTING------------------------

[nrows_test, ncols_test] = size(x_test);
x2 = ((x_test(:, 1) * svm.w(1, 1)) + svm.b) / svm.w(1, 2) * -1;

%{
ax + by +c = 0
y = (-ax -c) / b
%}

p_index = y_test == 1;
n_index = y_test == -1;

figure(1)
scatter(x_test(p_index, 1), x_test(p_index, 2), 'r')
hold on
scatter(x_test(n_index, 1), x_test(n_index, 2), 'b')
plot(x_test(:, 1), x2)
title('Decision Boundaries')
xlabel('x1')
ylabel('x2')
legend('1', '0', 'Predicted boundary')
saveas(gcf, 'decision.png');
