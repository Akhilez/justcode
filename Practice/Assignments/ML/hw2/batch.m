%----------------DATA CREATION-----------------------

nrows = 300;
a = 0.2;
b = 1.2;
x_train = (b-a).*randn(nrows,2) + a;
y_train = equation(x_train, 1, 2, -2);

%figure(3)
%scatter_classes(x_train, y_train);

% -------------------Perceptron--------------------

p = Perceptron;
p.weights = rand(1, 3);

% -------------------TRAINING--------------------
tic

epochs = 25;
metrics = p.train_batch(x_train, y_train, epochs, 0.001);

disp('training time:')
disp(toc);

% ------------------PLOTTING--------------------------

errors = metrics(1:epochs, 1);
weights_data = metrics(1:4, 2:end);

figure(1)
plot(1:length(errors), errors)
title('Batch Error vs Epochs')
xlabel('Epochs')
ylabel('Error')
saveas(gcf, 'figures/batch_error.png');

disp('weights');
disp(p.weights);

decision_y_pred = zeros(nrows, 1);
for i=1:length(weights_data)
  decision_y_pred(:, i) = line_equation(x_train(:, 1), weights_data(i, 1), weights_data(i, 2), weights_data(i, 3));
end
decision_y_real = line_equation(x_train(:, 1), 1, 2, -2);

scatter_classes_with_boundary(x_train, y_train, decision_y_pred, decision_y_real, 2);

% ------------------TESTING--------------------------

y_pred = p.test(x_train);

confusion_matrix = confusionmat(y_train, y_pred);
disp('confusion_matrix = ');
disp(confusion_matrix);

hit_rate = p.get_hit_rate(y_pred, y_train);
disp('Accuracy:');
disp(hit_rate);
