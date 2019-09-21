disp("helloooo")
all_data = csvread('../iris.txt');

% Use the CACC to discretize the floating point numbers.

[ discretized_data,discvalues,discscheme ] = cacc(all_data);

[nrows, ncols] = size(discretized_data);

rand_indices = randperm(nrows);

discretized_data = round(all_data);

train_set = discretized_data(rand_indices(1:fix(nrows*0.90)), :);
test_set = discretized_data(rand_indices(fix(nrows*0.90)+1:end), :);

[nodes, edges, parents] = run_down(train_set, [], [], []);

disp('nodes');
disp(nodes);
disp('edges');
disp(edges);
disp('parents');
disp(parents);

results = classify(test_set, nodes, edges, parents);

disp(results);
disp('reality: ');
disp(test_set(:, end));
