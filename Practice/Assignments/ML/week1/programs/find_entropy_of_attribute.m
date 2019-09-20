function entropy = find_entropy_of_attribute(dataset, attr_index)
    entropy = 0;
    
    % Find the number of rows of dataset
    [nrows_s, ncols_s] = size(dataset);
    
    %find unique values in attr_index col
    unique_vals = unique(dataset(:, attr_index));
    
    % for each unique value in attr
    for i = 1:size(unique_vals)
        
        % create a subset of dataset of those rows
        unique_val_indices = dataset(:, attr_index) == unique_vals(i);
        subset_data = dataset(unique_val_indices, :);
        
        % Find number of rows
        [nrows_a, ncols_a] = size(subset_data);
        
        % find entropy.
        entropy_a = find_entropy_of_set(subset_data);
        
        % multiply nrows/nrowsOfS * entropy
        entropy = entropy + nrows_a/nrows_s * entropy_a;
        
    end
    
    return;
    
end