function results = classify(test_set, nodes, edges, parents)

    [nrows, ncols] = size(test_set);
    
    results = zeros(nrows, 1);
    
    for i = 1:nrows 

        result = classify_instance(test_set(i, :), nodes, edges, parents);
        results(i, 1) = result;
        
    end
        
    return
end