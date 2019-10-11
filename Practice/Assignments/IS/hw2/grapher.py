import matplotlib.pyplot as plt


class Grapher:
    """
    The whole purpose of this class is just to create some subplots and store the x and y data.
    """

    def __init__(self):
        self.x = []
        self.y = []
        self.plt = plt

    def record(self, x_value, y_value):
        self.x.append(x_value)
        self.y.append(y_value)

    def plot(self, axis, title='Title', xlabel="x", ylabel="y", ylim=None, percentage_from_last=100, xticks=None,
             **plot_kwargs):
        starting_index = int((1 - percentage_from_last / 100) * len(self.x))
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        if ylim:
            axis.set_ylim(ylim)
        axis.set_title(title)
        if xticks:
            axis.set_xticks(xticks)
        return axis.plot(self.x[starting_index:], self.y[starting_index:], **plot_kwargs)

    def clear_data(self):
        self.__init__()

    @staticmethod
    def create_figure(num_rows, num_columns, figure_number, figsize=(16, 10)):
        return plt.subplots(num_rows, num_columns, constrained_layout=True, num=figure_number, figsize=figsize,
                            dpi=80)

    @staticmethod
    def show():
        plt.show()

    @staticmethod
    def save_figure(path):
        plt.savefig(path)
