%----------------DATA CREATION-----------------------

nrows = 300;
a = 0.2;
b = 1.2;
x_train = (b-a).*rand(nrows,2) + a;
y_train = equation(x_train);

% scatter_classes(x_train, y_train)

x_test = (b-a).*rand(50,2) + a;
y_test = equation(x_test);

% -------------------Perceptron--------------------

p = Perceptron;
p.weights = rand(1, 3);

disp('weights');
disp(p.weights);


% -------------------TRAINING--------------------

errors = p.train_incremental(x_train, y_train, 500, 0.001);

plot(1:length(errors), errors);

disp('weights');
disp(p.weights);


% ------------------TESTING--------------------------

y_pred = p.test(x_test);

% scatter_classes(x_test, y_pred);

confusion_matrix = confusionmat(y_test, y_pred);
disp(confusion_matrix);

