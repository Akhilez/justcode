% ------------------GET DATA-------------------------
%%{

all_data = csvread('hayes-roth.data');

[nrows, ncols] = size(all_data);

x = all_data(:, 1:end-1);
y = all_data(:, end);

for i = 1:ncols-1
  V = x(:,i);
  maxV = max(V(:));
  minV = min(V(:));
  x(:,i) = (V - minV) / (maxV - minV);
end

y(y == 1) = -1;
y(y == 2) = 0;
y(y == 3) = 1;

split_index = floor(nrows * 0.8);

x_train = x(1:split_index, :);
y_train = y(1:split_index, :);
x_test = x(split_index:end, :);
y_test = y(split_index:end, :);


%}
% -------------------GENERATING DATA------------------
%{
n = 4;
gd1 = GDEquation(1:4);

nrows = 10;
x = rand(nrows, n-1);

y = zeros(nrows, 1);
for i = 1:nrows
  y(i) = gd1.equation(x(i, :));
end
%}

%-------------------TRAINING----------------------
%%{

weights = rand(ncols, 1);
gd2 = GDEquation(weights);

errors = gd2.train(x_train, y_train, 400, 0.001);
disp('error = ');
disp(errors(end));

disp('learned weights')
disp(gd2.weights);

figure(1)
plot(25:length(errors), errors(25:end))
title('Problem 3 Error')
xlabel('Epochs')
ylabel('Training Error')
saveas(gcf, 'figures/problem3.png');

y_pred = gd2.test(x_test);

mat = confusionmat(y_test, y_pred);
disp('Confusion matrix = ');
disp(mat);

% disp([y_test, y_pred]);

hit_rate = gd2.get_hit_rate(y_pred, y_test);
disp('accuracy = ');
disp(hit_rate);

%}
