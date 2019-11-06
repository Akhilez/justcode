import matplotlib.pyplot as plt


class Grapher:

    fig_no = 1

    def __init__(self):
        self.x = []
        self.y = []
        self.plt = plt

    def record(self, x_value, y_value):
        self.x.append(x_value)
        self.y.append(y_value)

    def plot(self, axis, title='Title', xlabel="x", ylabel="y", ylim=None, percentage_from_last=100, xticks=None, **plot_kwargs):
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
        return plt.subplots(num_rows, num_columns, num=figure_number, figsize=figsize, dpi=80, constrained_layout=True)

    @staticmethod
    def show():
        plt.show()

    @staticmethod
    def save_figure(path):
        plt.savefig(path)

    @staticmethod
    def get_new_fig_number():
        fig_no = Grapher.fig_no
        Grapher.fig_no += 1
        return fig_no

    @staticmethod
    def plot_generic(x, y, title, x_label, y_label, fig_name):
        grapher = Grapher()
        fig, axs = grapher.create_figure(1, 1, Grapher.get_new_fig_number(), figsize=(6, 5))

        for i in range(len(x)):
            grapher.record(x[i], y[i])

        grapher.plot(axs, title=title, xlabel=x_label, ylabel=y_label, xticks=grapher.x)
        grapher.save_figure(f'figures/{fig_name}.png')
        grapher.show()
