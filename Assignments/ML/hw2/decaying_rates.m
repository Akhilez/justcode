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

errors = p.train_decaying_lr(x_train, y_train, 25, 0.01, 0.6);

% ------------------PLOTTING--------------------------

figure(1)
plot(1:length(errors), errors)
title('Decaying rate Epochs vs Error')
xlabel('Epochs')
ylabel('Error')
saveas(gcf, 'figures/decaying_rates_error.png');

disp('weights');
disp(p.weights);

% ------------------TESTING--------------------------

y_pred = p.test(x_train);

confusion_matrix = confusionmat(y_train, y_pred);
disp('confusion_matrix = ');
disp(confusion_matrix);

hit_rate = p.get_hit_rate(y_pred, y_train);
disp('Accuracy:');
disp(hit_rate);
