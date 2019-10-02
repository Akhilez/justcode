all_data = csvread('iris.txt');

% Use the CACC to discretize the floating point numbers.

[ discretized_data,discvalues,discscheme ] = cacc(all_data);

[nrows, ncols] = size(discretized_data);

disp('Starting Decision Tree creations');
for i = (1:5)
    
    disp('Iteration #:');
    disp(i);
    
    rand_indices = randperm(nrows);
    
    train_set = discretized_data(rand_indices(1:fix(nrows*0.70)), :);
    test_set = discretized_data(rand_indices(fix(nrows*0.70)+1:end), :);

    [nodes, edges, parents] = run_down(train_set, [], [], []);

    disp('nodes_names');
    disp((1:length(nodes)));
    disp('node_values');
    disp(nodes);
    disp('edges');
    disp(edges);
    disp('parents');
    disp(parents);

    results = classify(test_set, nodes, edges, parents);

    accuracy = classperf(results, test_set(:, end));
    disp('accuracy in percentage: ')
    disp(accuracy.CorrectRate * 100);
    
    confusion_matrix = confusionmat(test_set(:, end), results);
    disp('confusion matrix = ');
    disp(confusion_matrix);

end