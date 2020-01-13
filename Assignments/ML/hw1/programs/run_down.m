function [nodes, edges, parents] = run_down(available_dataset, nodes, edges, parents, attr_names)

    [~, ncols] = size(available_dataset);

    % Initializing the outputs
    if size(edges) == 0
        edges = -1;
        parents = -1;
        attr_names = 1:ncols;
    end

    % If there are no attributes
    if ncols == 1
        % Add leaf node.
        nodes = [nodes, mode(available_dataset)];
        return;
    end
    
    % If all the values in target are the same, add that as leaf
    unique_target_vals = unique(available_dataset(:, end));
    if length(unique_target_vals) == 1
        nodes = [nodes, unique_target_vals(1)];
        return;
    end
    
    % Finding entropy of the entire dataset.
    entropy_s = find_entropy_of_set(available_dataset);
    
    % Find Information Gain
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
    [~, max_index] = max(gains);
    max_index = max_index(1);
    
    % Add this attribute to the tree
    nodes = [nodes, attr_names(max_index)];
    current_node_index = length(nodes);
    
    % For each unique value of this attribute
    unique_vals = unique(available_dataset(:, max_index));
    for i = 1: size(unique_vals)
        
        % Create a new available_dataset by removing it's rows and col.
        subtracted_set = available_dataset(:, :);
        val_indices = subtracted_set(:, max_index) == unique_vals(i);
        subtracted_set = subtracted_set(val_indices, :);
        subtracted_set(:, max_index) = [];
        
        % If all rows are covered, then continue;
        [nrows_sub, ~] = size(subtracted_set);
        if nrows_sub == 0
            continue;
        end
        
        % Create the edge with this value and mark the parent
        edges = [edges, unique_vals(i)];
        parents = [parents, current_node_index];
        
        % Preserving the attribute indices even in reduction
        next_attr_names = attr_names;
        next_attr_names(max_index) = [];
        
        % Call run_down with this subtracted dataset.
        [nodes, edges, parents] = run_down(subtracted_set, nodes, edges, parents, next_attr_names);

    end
    
    return
    
end