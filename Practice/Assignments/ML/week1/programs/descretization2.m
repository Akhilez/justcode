all_data = csvread('iris.txt');

% Use the CACC to discretize the floating point numbers.

[ discdata,discvalues,discscheme ] = cacc(all_data);

disp('Printing the discrete data below:')

disp(discdata)
