all_data = csvread('../iris.txt');

% Using round function to round off the floating numbers into integers.
discrete_data = round(predictors);

disp(discrete_data)
