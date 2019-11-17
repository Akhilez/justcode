neg_points = [0.4, 0.3; 0.3, 0.5; 0.1, 0.6; 0.3, 0.2];
pos_points = [0.7, 0.4; 0.9, 0.3; 0.8, 0.5; 0.55, 0.6; 0.6, 0.8];

sample_eq = "y = -2x + 1.4";

y_neg = [-1, -1, -1, -1];
y_pos = [1, 1, 1, 1, 1];

y = cat(2, y_neg, y_pos);
x = cat(1, neg_points, pos_points);

svm = Smo(x, y);

svm.train();

