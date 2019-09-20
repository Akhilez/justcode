disp("helloooo")
all_data = csvread('../iris.txt');

% Use the CACC to discretize the floating point numbers.

[ discretized_data,discvalues,discscheme ] = cacc(all_data);

[nodes, edges, parents] = run_down(discretized_data, [], [], []);

disp('nodes');
disp(nodes);
disp('edges');
disp(edges);
disp('parents');
disp(parents);