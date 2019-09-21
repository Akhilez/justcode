function result = classify_instance(x, nodes, edges, parents)

    result = -1;

    node_i = 1;
    
    i = 1;
    while i<10000
        i = i + 1;
        
        test_value = x(1, nodes(node_i));
        
        found = 0;
        % find all node_index in parents.
        for j = 1:length(parents)
            if parents(j) == node_i
               
                % Check if the test value matches with the edge value.
                if test_value == edges(j)
                    % Whichever edge it does, get that edge's node and continue.
                    node_i = j;
                    found = 1;
                    break;
                end
            end
        end
        
        % if node index is not present in parents, then return node(i)
        if found == 0
            result = nodes(node_i);
            return;
        end
    
    end
    
    

end