import numpy as np
import matplotlib.pyplot as plt


class SpikingNeuron:
    def __init__(self):
        self.a = 0.02
        self.b = 0.25
        self.c = -65
        self.d = 6
        self.voltage = -64
        self.recovery = self.b * self.voltage
        self.graph = Grapher()

    def simulate_input(self, input_voltage_function, time_span, time_step):
        for time_i in np.arange(0, time_span + time_step, time_step):
            current_voltage = input_voltage_function(time_i)

            # Applying the input voltage
            self.voltage += time_step * (
                    0.04 * self.voltage ** 2 + 5 * self.voltage + 140 - self.recovery + current_voltage)
            self.recovery += time_step * self.a * (self.b * self.voltage - self.recovery)

            if self.voltage > 30:
                # Recording the voltage
                self.graph.record(time_i, 30)

                # Refractory period
                self.voltage = self.c
                self.recovery += self.d
            else:
                self.graph.record(time_i, self.voltage)


class Grapher:

    def __init__(self):
        self.x = []
        self.y = []

    def record(self, x_value, y_value):
        self.x.append(x_value)
        self.y.append(y_value)

    def plot(self, axis, title='Regular Spiking'):
        axis.set_xlabel("time step")
        axis.set_ylabel("V_m")
        axis.set_ylim((-90, 40))
        axis.set_title(title)
        axis.plot(self.x, self.y)

    def clean(self):
        self.__init__()

    @staticmethod
    def enable_subplot(num_rows, num_columns, figure_number):
        return plt.subplots(num_rows, num_columns, constrained_layout=True, num=figure_number, figsize=(16, 10), dpi=80)


def regular_spiking():
    neuron = SpikingNeuron()
    neuron.simulate_input(lambda time: 1 if time > 50 else 0, time_span=1000, time_step=0.25)
    fig, axs = neuron.graph.enable_subplot(num_rows=1, num_columns=1, figure_number=1)
    neuron.graph.plot(axs[0][0])
    plt.show()


def observations_for_81_inputs():
    voltage_step = 0.25
    num_rows, num_cols = (4, 5)
    for fig_i in range(4):
        fig, axs = Grapher.enable_subplot(num_rows, num_cols, fig_i)
        fig.suptitle(f'Regular Spiking Neuron Figure {fig_i + 1}', fontsize=16)
        for row_i in range(len(axs)):
            for col_j in range(len(axs[row_i])):
                input_voltage = voltage_step * (row_i * num_cols + col_j) + (fig_i * num_rows * num_cols * voltage_step)
                neuron = SpikingNeuron()
                neuron.simulate_input(lambda time: input_voltage, time_span=1000, time_step=0.25)
                neuron.graph.plot(axs[row_i][col_j], title=f"Input = {input_voltage}")
                neuron.graph.clean()
        plt.savefig(f'figures/{fig_i}')
        # plt.show()


observations_for_81_inputs()
# regular_spiking()
