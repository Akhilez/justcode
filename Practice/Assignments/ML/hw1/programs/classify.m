% This function will parse the decision tree and classify the test data

function results = classify(test_set, nodes, edges, parents)

    [nrows, ~] = size(test_set);
    
    results = zeros(nrows, 1);
    
    % For each row in the test data
    for row = 1:nrows 
        node_i = 1;

        i = 1;
        while i<10000
            i = i + 1;

            test_value = test_set(row, nodes(node_i));

            found = 0;
            % find all node_index values in parents.
            for j = 1:length(parents)
                if parents(j) == node_i

                    % Check if the test value matches with the edge value.
                    if test_value == edges(j)
                        % Whichever edge it matches, get that edge's node and continue.
                        node_i = j;
                        found = 1;
                        break;
                    end
                end
            end

            % if node index is not present in parents, then we reached leaf node.
            if found == 0
                result = nodes(node_i);
                break;
            end

        end
        
        results(row, 1) = result;
        
    end
        
    return
end