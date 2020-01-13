function entropy = find_entropy_of_set(dataset)
    entropy = 0;
    
    [nr, nc] = size(dataset);
    
    if nc < 1
        disp('No columns to find entropy');
        return;
    end

    target_c = dataset(:, nc);

    unique_val = unique(target_c);
    unique_val_size = size(unique_val);

    num_vals = zeros(unique_val_size);

    % Getting the count of each unique value
    for i = 1:nr
        cur_t = target_c(i);
        for j = 1:unique_val_size
            if cur_t == unique_val(j)
                num_vals(j) = num_vals(j) + 1;
            end
        end
    end

    for i = 1:unique_val_size
        proportion = num_vals(i) / nr;
        entropy = entropy + (proportion * log2(proportion));
    end
    entropy = entropy * -1;

    return;
end