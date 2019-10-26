% -------------------GENERATING DATA------------------
n = 4;
gd1 = GDEquation(1:4);

nrows = 10;
x = rand(nrows, n-1);

y = zeros(nrows, 1);
for i = 1:nrows
  y(i) = gd1.equation(x(i, :));
end

%-------------------TRAINING----------------------

weights = rand(n, 1);
gd2 = GDEquation(weights);

errors = gd2.train(x, y, 50, 0.1);
disp('error = ');
disp(errors(end));

disp('real weights');
disp(gd1.weights);

disp('learned weights')
disp(gd2.weights);

plot(1:length(errors), errors);
