import math

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
        self.vt_graph = Grapher()
        self.spike_times = []

    def simulate_input(self, input_voltage_function, time_span, time_step):
        for time_i in np.arange(0, time_span + time_step, time_step):
            current_voltage = input_voltage_function(time_i)

            # Applying the input voltage
            self.voltage += time_step * (
                    0.04 * self.voltage ** 2 + 5 * self.voltage + 140 - self.recovery + current_voltage)
            self.recovery += time_step * self.a * (self.b * self.voltage - self.recovery)

            if self.voltage > 30:
                # Recording the voltage
                self.vt_graph.record(time_i, 30)

                # Recording the time of spike
                self.spike_times.append(time_i)

                # Refractory period
                self.voltage = self.c
                self.recovery += self.d
            else:
                self.vt_graph.record(time_i, self.voltage)

    def get_spiking_rate(self, time_span=1000, percentage_from_last=80):
        discarded_steps = int((1 - percentage_from_last / 100) * time_span)
        i = 0
        while len(self.spike_times) > i and self.spike_times[i] < discarded_steps:
            i += 1
        required_spikes = self.spike_times[i:]
        return len(required_spikes) / (time_span - discarded_steps)


class Grapher:

    def __init__(self):
        self.x = []
        self.y = []

    def record(self, x_value, y_value):
        self.x.append(x_value)
        self.y.append(y_value)

    def plot(self, axis, title='Title', xlabel="x", ylabel="y", ylim=None):
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        if ylim:
            axis.set_ylim(ylim)
        axis.set_title(title)
        axis.plot(self.x, self.y)

    def clear_data(self):
        self.__init__()

    @staticmethod
    def create_figure(num_rows, num_columns, figure_number, figsize=(16, 10)):
        return plt.subplots(num_rows, num_columns, constrained_layout=True, num=figure_number, figsize=figsize, dpi=80)


def regular_spiking():
    neuron = SpikingNeuron()
    neuron.simulate_input(lambda time: 1 if time > 50 else 0, time_span=1000, time_step=0.25)
    fig, axs = neuron.vt_graph.create_figure(num_rows=1, num_columns=1, figure_number=1)
    neuron.vt_graph.plot(axs[0][0])
    plt.show()


def simulate_constant_inputs_range(min_input, max_input, voltage_step, time_span=1000, time_step=0.25, num_rows=1,
                                   num_cols=1, save_figures=False):
    num_figures = math.ceil((max_input - min_input) / voltage_step / num_rows / num_rows)
    rt_graph = Grapher()
    for fig_i in range(num_figures):
        fig, axs = Grapher.create_figure(num_rows, num_cols, fig_i)
        axs = axs.flatten()
        fig.suptitle(f'Regular Spiking Neuron Figure {fig_i + 1}', fontsize=16)
        for i in range(len(axs)):
            input_voltage = min_input + voltage_step * i + (fig_i * num_rows * num_cols * voltage_step)
            if input_voltage > max_input:
                break
            neuron = SpikingNeuron()
            neuron.simulate_input(lambda time: input_voltage, time_span=time_span, time_step=time_step)
            neuron.vt_graph.plot(axs[i], title=f"Input = {input_voltage}", xlabel="time step", ylabel="V_m", ylim=(-90, 40))
            spike_rate = neuron.get_spiking_rate()
            rt_graph.record(input_voltage, spike_rate)
            neuron.vt_graph.clear_data()
        if save_figures:
            plt.savefig(f'figures/vt_graph_{fig_i}')

    rt_graph.plot(rt_graph.create_figure(1, 1, 1)[1], title="Firing rate against input", xlabel="Input",
                  ylabel="Firing Rate")
    if save_figures:
        plt.savefig('figures/rt_graph')

    plt.show()


def simulate_constant_inputs(inputs, time_span=1000, time_step=0.25, save_figures=False):
    fig, axs = Grapher.create_figure(len(inputs), 1, 1, figsize=(6, 10))
    fig.suptitle(f'Regular Spiking Neuron Figure 6', fontsize=16)
    for i in range(len(inputs)):
        input_voltage = inputs[i]
        neuron = SpikingNeuron()
        neuron.simulate_input(lambda time: input_voltage, time_span=time_span, time_step=time_step)
        neuron.vt_graph.plot(axs[i], title=f"Input = {input_voltage}", xlabel="time step", ylabel="V_m", ylim=(-90, 40))
        neuron.vt_graph.clear_data()
    if save_figures:
        plt.savefig(f'figures/vt_graph_6')
    plt.show()


'''
simulate_constant_inputs(voltage_step=0.25,
                         min_input=0,
                         max_input=20,
                         num_rows=4,
                         num_cols=5
                         )
'''

simulate_constant_inputs(inputs=[1, 5, 10, 15, 20], save_figures=False)


# regular_spiking()
