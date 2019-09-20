function [nodes, edges, parents] = run_down(available_dataset, nodes, edges, parents, attr_names)
    [nrows, ncols] = size(available_dataset);
    if ncols == 1
        disp('no attribute left');
        % Add leaf node.
        nodes = [nodes, mode(available_dataset)];
        return;
    end
    
    entropy_s = find_entropy_of_set(available_dataset);
    
    gains = zeros(1, ncols - 1);
    
    % For each attribute
    for i = 1:(ncols - 1)
        
        % get the entropy of that column.
        entropy = find_entropy_of_attribute(available_dataset, i);
        
        % Find the information gain.
        gain = entropy_s - entropy;
        
        % Store information gain
        gains(1, i) = gain;
        
    end
    
    % Find the max gain attribute
    [max_val, max_index] = max(gains);
    max_index = max_index(1);
    
    if size(edges) == 0
        edges = -1;
        parents = -1;
        attr_names = 1:ncols;
    end
    
    % Add this attribute to the tree
    nodes = [nodes, attr_names(max_index)];
    current_node_index = length(nodes);
    
    % For each unique value of this attribute
    unique_vals = unique(available_dataset(:, max_index));
    for i = 1: size(unique_vals)
        % Create a new available_dataset by removing it's rows and col.
        subtracted_set = available_dataset(:, :);
        val_indices = subtracted_set(:, max_index) ~= unique_vals(i);
        subtracted_set = subtracted_set(val_indices, :);
        subtracted_set(:, max_index) = [];
        
        % If all rows are covered, then continue;
        [nrows_sub, ncols_sub] = size(subtracted_set);
        if nrows_sub == 0
            continue;
        end
        
        % Call run_down with this new dataset.
        edges = [edges, unique_vals(i)];
        parents = [parents, current_node_index];
        next_attr_names = attr_names;
        next_attr_names(max_index) = [];
        % next_node_index = length(nodes) + 1;
        [nodes, edges, parents] = run_down(subtracted_set, nodes, edges, parents, next_attr_names);
        % attr_names(attr_names == nodes(next_node_index)) = [];
    end
    
    return
    
end