from data_manager import DataManager
from grapher import Grapher
import matplotlib.pyplot as plt


class PerformanceAnalyzer:
    figure_number = 1

    def __init__(self):
        self.metrics = None

    def run(self):
        data_manager = DataManager('hw2_dataProblem.txt')
        data = data_manager.get_data()
        data = data_manager.get_column_wise_rescaled_data(data)

        self.metrics = {'knn': [], 'neighbourhood': [], 'perceptron': []}

        for i in range(9):
            x_train, y_train, x_test, y_test = data_manager.test_train_split(data)

            self.metrics['knn'].append(self.run_knn(x_train, y_train, x_test, y_test))
            self.metrics['neighbourhood'].append(self.run_neighbourhood(x_train, y_train, x_test, y_test))
            self.metrics['perceptron'].append(self.run_perceptron(x_train, y_train, x_test, y_test))

        # self.plot_trails()
        # self.plot_average_performance()
        self.plot_training_error()
        self.plot_mean_training_error()
        self.plot_knn_decision_boundary()
        self.plot_best_neighbourhood_boundary()
        self.plot_perceptron_boundary()

    def run_knn(self, x_train, y_train, x_test, y_test):
        return {'test': {'sensitivity': 1, 'specificity': 2, 'PPV': 3, 'NPV': 4, 'hit-rate': 0.91}}

    def run_neighbourhood(self, x_train, y_train, x_test, y_test):
        return {'test': {'sensitivity': 0, 'specificity': 0, 'PPV': 0, 'NPV': 0, 'hit-rate': 0.92}}

    def run_perceptron(self, x_train, y_train, x_test, y_test):
        return {'test': {'sensitivity': 1, 'specificity': 2, 'PPV': 3, 'NPV': 4, 'hit-rate': 0.93},
                'train': {'sensitivity': 1, 'specificity': 2, 'PPV': 3, 'NPV': 4},
                'train-error': {'epocs': [1, 10, 20], 'error': [0.9, 0.8, 0.5]}}

    def plot_trails(self):
        """
        Performance on Individual Trials: For the perceptron algorithm, four bar plots with nine groups
        of two bars each. Each plot will be for one of the metrics (sensitivity, specificity, PPV and NPV) ,
        with each pair of bars showing the value of the metric for final training and final testing on one trial.
        The training bar should be on the left, the testing on the right.
        For the other two algorithms, four bar plots with nine bars each. Each plot will be for one of the metrics
        sensitivity, specificity, PPV and NPV, with each bar showing the value of the metric for testing on one
        trial.
        Thus, in all, you will have twelve total plots - four for each classifier. The four graphs for each classifier
        should be grouped into one figure with either four panels in one row or a 2×2 arrangement. Thus, this
        part will have a total of three figures, each with four panels.
        """
        self.plot_trails_single_bars(self.metrics['knn'], 'KNN Classifier')
        self.plot_trails_single_bars(self.metrics['neighbourhood'], 'Neighbourhood Classifier')
        self.plot_trails_double_bars(self.metrics['perceptron'], 'Perceptron Classifier')

    @staticmethod
    def plot_trails_double_bars(metrics, title):
        fig, axs = Grapher.create_figure(1, 4, PerformanceAnalyzer.figure_number, (16, 4))
        PerformanceAnalyzer.figure_number += 1
        graphs = [Grapher() for i in range(4)]
        trails_count = [i + 1 for i in range(len(metrics))]

        for i in range(len(metrics)):
            graphs[0].record(metrics[i]['train']['sensitivity'], metrics[i]['test']['sensitivity'])
            graphs[1].record(metrics[i]['train']['specificity'], metrics[i]['test']['specificity'])
            graphs[2].record(metrics[i]['test']['PPV'], metrics[i]['test']['PPV'])
            graphs[3].record(metrics[i]['train']['NPV'], metrics[i]['test']['NPV'])

        for i in range(4):
            axs[i].bar(trails_count, graphs[i].x, color='b', label='Train', width=0.25)
            axs[i].bar([t + 0.25 for t in trails_count], graphs[i].y, color='g', label='Test', width=0.25)
            axs[i].set_xticks(trails_count)
            axs[i].set_xlabel('Trails')
            axs[i].legend(['Train', 'Test'])

        fig.suptitle(title)

        axs[0].set_title('sensitivity')
        axs[1].set_title('specificity')
        axs[2].set_title('ppv')
        axs[3].set_title('npv')

    @staticmethod
    def plot_trails_single_bars(metrics, title):
        fig, axs = Grapher.create_figure(1, 4, PerformanceAnalyzer.figure_number, (16, 4))
        PerformanceAnalyzer.figure_number += 1
        graphs = [Grapher() for i in range(4)]
        for i in range(len(metrics)):
            graphs[0].record(i + 1, metrics[i]['test']['sensitivity'])
            graphs[1].record(i + 1, metrics[i]['test']['specificity'])
            graphs[2].record(i + 1, metrics[i]['test']['PPV'])
            graphs[3].record(i + 1, metrics[i]['test']['NPV'])

        for i in range(4):
            axs[i].bar(graphs[i].x, graphs[i].y)
            axs[i].set_xticks(graphs[i].x)
            axs[i].set_xlabel('Trails')

        fig.suptitle(title)

        axs[0].set_title('sensitivity')
        axs[1].set_title('specificity')
        axs[2].set_title('ppv')
        axs[3].set_title('npv')

    def plot_average_performance(self):
        """
        Average Performance: A table giving the mean values of sensitivity, specificity, PPV, NPV, and
        hit-rate for each algorithm on testing data averaged over all 9 trials, along with the standard deviation
        for each case. The standard deviations will be indicated as ± values after the mean, e.g., 0.93 ± 0.03.
        Each table will have five data rows – one for each metric – and three data columns - one for each
        algorithm.
        """
        sensitivities = [[], [], []]
        specificities = [[], [], []]
        ppvs = [[], [], []]
        npvs = [[], [], []]
        hit_rates = [[], [], []]
        for i in range(len(self.metrics['knn'])):
            sensitivities[0].append(self.metrics['knn'][i]['test']['sensitivity'])
            sensitivities[1].append(self.metrics['neighbourhood'][i]['test']['sensitivity'])
            sensitivities[2].append(self.metrics['perceptron'][i]['test']['sensitivity'])

            specificities[0].append(self.metrics['knn'][i]['test']['specificity'])
            specificities[1].append(self.metrics['neighbourhood'][i]['test']['specificity'])
            specificities[2].append(self.metrics['perceptron'][i]['test']['specificity'])

            ppvs[0].append(self.metrics['knn'][i]['test']['PPV'])
            ppvs[1].append(self.metrics['neighbourhood'][i]['test']['PPV'])
            ppvs[2].append(self.metrics['perceptron'][i]['test']['PPV'])

            npvs[0].append(self.metrics['knn'][i]['test']['NPV'])
            npvs[1].append(self.metrics['neighbourhood'][i]['test']['NPV'])
            npvs[2].append(self.metrics['perceptron'][i]['test']['NPV'])

            hit_rates[0].append(self.metrics['knn'][i]['test']['hit-rate'])
            hit_rates[1].append(self.metrics['neighbourhood'][i]['test']['hit-rate'])
            hit_rates[2].append(self.metrics['perceptron'][i]['test']['hit-rate'])

        sensitivities_avg = [self.get_avg_with_std(x) for x in sensitivities]
        specificities_avg = [self.get_avg_with_std(x) for x in specificities]
        ppvs_avg = [self.get_avg_with_std(x) for x in ppvs]
        npvs_avg = [self.get_avg_with_std(x) for x in npvs]
        hit_rates_avg = [self.get_avg_with_std(x) for x in hit_rates]

        data = [sensitivities_avg, specificities_avg, ppvs_avg, npvs_avg, hit_rates_avg]
        row_labels = ['sensitivity', 'specificity', 'PPV', 'NPV', 'hit-rate']
        col_labels = ['knn', 'neighbourhood', 'perceptron']

        fig, ax = plt.subplots(figsize=(7, 3))

        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')

        ax.table(data, colLabels=col_labels, rowLabels=row_labels, loc='center')

    @staticmethod
    def get_avg_with_std(data):
        avg = round(sum(data) / len(data), 2)
        # TODO: Get standard deviation
        std = 0.23
        return f'{avg} ± {std}'

    def plot_training_error(self):
        """
        Trial-Wise Training Error Time-Series for the Perceptrons: For the perceptron case only:
        graphs showing the error (1 - hit-rate) on the training set plotted against epoch over the duration of
        training for the nine trials . Each plot will start with the initial error and plot the error at every tenth
        epoch. You should plot all nine curves on the same graph. You should choose graph properties (line
        thicknesses, colors, etc.) for maximum clarity.
        """
        pass

    def plot_mean_training_error(self):
        """
        Mean Training Error for the Perceptron: For the perceptron case only: a graph showing the
        mean training error averaged over the 9 trials plotted against time. This graph will have only one
        plotted curve with a datapoint every 10 epochs, where each point of the curve is the average value
        of the 9 points at the corresponding time in the graph above. At each plotted point, put error bars
        indicating the ± standard deviation over the 9 trials.
        """
        pass

    def plot_knn_decision_boundary(self):
        """
        Best k-NN Decision Boundary: For k-NN only: a plot of the data in the feature space indicating
        the decision boundaries found by the classifier in the best trial. This will involve sampling the feature
        space as we discussed in class, and will only give an approximate boundary.
        """
        pass

    def plot_best_neighbourhood_boundary(self):
        """
        Best Neighborhood Classifier Decision Boundary: For neighborhood-based classifier only: a
        plot of the data in the feature space indicating the decision boundaries found by the classifier in the
        best trial. The approach used will be the same as for k-NN, i.e., sample the feature space to get
        classification.
        """
        pass

    def plot_perceptron_boundary(self):
        """
        Perceptron Decision Boundary: For the perceptron classifier only: Take the perceptron (out of
        the 9) that had the best performance, and plot the decision boundary it found in the feature space.
        Your figure size and L-P coordinate ranges for this figure and the previous two figures should be the
        same so the three figures can be compared.
        """
        pass


def main():
    performance = PerformanceAnalyzer()
    performance.run()


if __name__ == "__main__":
    main()
