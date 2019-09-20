disp("helloooo")
all_data = csvread('../iris.txt');
discretized_data = round(all_data);
[nodes, edges, parents] = run_down(discretized_data, [], [], []);

disp('nodes');
disp(nodes);
disp('edges');
disp(edges);
disp('parents');
disp(parents);