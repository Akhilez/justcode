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

disp('weights');
disp(p.weights);


% -------------------TRAINING--------------------

metrics = p.train_batch(x_train, y_train, 250, 0.0001);

weights_arr = metrics(:, 1:end-1);
errors = metrics(:, end);

% ------------------PLOTTING--------------------------

figure(1)
plot(1:length(errors), errors)
title('Training Error vs Epochs')
xlabel('Epochs')
ylabel('Training MSE')
saveas(gcf, 'figures/error.png');

disp('weights');
disp(p.weights);

decision_y_pred = line_equation(x_train(:, 1), p.weights(1), p.weights(2), p.weights(3));
decision_y_real = line_equation(x_train(:, 1), 1, 2, -2);

scatter_classes_with_boundary(x_train, y_train, x_train(:, 1), decision_y_pred, x_train(:, 1), decision_y_real, 2);

% ------------------TESTING--------------------------

y_pred = p.test(x_train);

confusion_matrix = confusionmat(y_train, y_pred);
disp(confusion_matrix);

