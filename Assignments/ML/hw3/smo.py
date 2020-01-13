import numpy as np


class Smo:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.n_rows_x, self.n_cols_x = self.x.shape
        self.epsilon = 0.5
        self.C = 2
        self.alpha = self.get_init_alpha()
        self.b = 0
        self.w = self.get_init_w()

    def get_init_w(self):
        w_sum = np.zeros(self.x.shape)
        for i in range(self.n_rows_x):
            w_sum[i] += self.alpha[i] * self.y[i] * self.x[i]
        return w_sum

    def get_error_i(self, i):
        sum_term = 0
        for j in range(self.n_rows_x):
            sum_term += self.alpha[j] * self.y[j] * self.get_kernel_value(i, j) + self.b
        return sum_term - self.y[i]

    def get_init_alpha(self):
        pos_count, neg_count = 0, 0
        for yi in self.y:
            if yi < 0:
                neg_count += 1
            else:
                pos_count += 1
        max_count = max((pos_count, neg_count))
        min_count = min((pos_count, neg_count))
        is_max_positive = max_count == pos_count
        min_alphas = np.random.uniform(0, self.C, min_count)
        remaining_alphas = np.random.uniform(0, min(min_alphas), max_count - min_count)
        reduction = sum(remaining_alphas) / min_count
        max_alphas = np.concatenate(((min_alphas - reduction), remaining_alphas), axis=0)
        alphas = []
        pos_alphas = max_alphas if is_max_positive else min_alphas
        neg_alphas = min_alphas if is_max_positive else max_alphas
        pos_i, neg_i = 0, 0
        for i in range(self.n_rows_x):
            if self.y[i] < 0:
                alphas.append(neg_alphas[neg_i])
                neg_i += 1
            else:
                alphas.append(pos_alphas[pos_i])
                pos_i += 1

        return np.array(alphas)

    def get_kernel_value(self, i, j):
        return self.x[i].dot(self.x[j])


def main():
    neg_points = ((0.4, 0.3), (0.3, 0.5), (0.1, 0.6), (0.3, 0.2))
    pos_points = ((0.7, 0.4), (0.9, 0.3), (0.8, 0.5), (0.55, 0.6), (0.6, 0.8))
    sample_eq = "y = -2x + 1.4"

    y_neg = [-1 for i in range(len(neg_points))]
    y_pos = [1 for i in range(len(pos_points))]

    x = neg_points + pos_points
    y = y_neg + y_pos

    svm = Smo(x, y)


if __name__ == "__main__":
    main()
