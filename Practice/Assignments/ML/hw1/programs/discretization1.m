all_data = csvread('iris.txt');

% Using round function to round off the floating numbers into integers.
discrete_data = round(all_data);

disp('Printing the discrete data below:')

disp(discrete_data)
