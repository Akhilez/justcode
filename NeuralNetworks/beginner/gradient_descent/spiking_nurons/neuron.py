"""
Author: Akhilez.com
Date: 9/13/19
Description: This file contains two classes.
1. Spiking Neuron class:
  - Takes care of the underlying voltage equation
  - It can invoke an input stimulation for given duration.
2. Grapher:
  - This class stores x and y values overtime and produces graphs for them.
"""
import math
import numpy as np
import matplotlib.pyplot as plt


class SpikingNeuron:
    TYPES = {'regular': {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 6}}

    def __init__(self, neuron_type='regular'):
        """
        A Spiking Neuron model that can act as a regular neuron as of now, but has room to behave differently.
        :param neuron_type: String: default is 'regular', will be adding more types if needed.
        """
        self.a = SpikingNeuron.TYPES[neuron_type]['a']
        self.b = SpikingNeuron.TYPES[neuron_type]['b']
        self.c = SpikingNeuron.TYPES[neuron_type]['c']
        self.d = SpikingNeuron.TYPES[neuron_type]['d']
        self.voltage = -64
        self.recovery = self.b * self.voltage
        self.vt_graph = Grapher()
        self.spike_times = []

    def stimulate_input(self, input_time_function, time_span, time_step, record_values=True, yield_values=False):
        """
        This is the heart of the Spiking Neuron class.
        The input voltage to the neuron is passed as a time function. So input can vary with time, but in all but
        the demo case, the input does not change with time.
        This is a generator function, but doesn't necessarily return an iterator. It depends of yield_values param.
        Each iteration is considered as a mini-step. For example, for the time_span of 1000 ms and 0.25 time_step,
        there would be 4001 mini steps.
        For each iteration, the membrane potential is calculated.
        If record_values is enabled, then the membrane potentials at each time step is recorded and
        if there was a spike, then the spike time will be recorded in self.spike_times list.
        :param input_time_function: Lambda [(time:Float) => voltage: Float]: input function with time variable.
        :param time_span: Integer: The total time the neuron would be supplied with input voltage.
        :param time_step: Float: The mini-step interval at which the voltage would be calculated.
        :param record_values: Boolean: Records the voltages and spikes if True.
        :param yield_values: Boolean: Defines whether the voltage needs to be yielded at each iteration or not.
        :return: Yields the voltage at each time mini-step if yield_values is enabled, else None.
        """
        # Iteration for each time mini-step starts here.
        for time_i in np.arange(0, time_span + time_step, time_step):

            # Get the input voltage that will be supplied from the time voltage function.
            input_voltage = input_time_function(time_i)

            # Apply the voltage to the neuron.
            self.apply_input_voltage(time_step, input_voltage)

            # Checking whether the applied voltage resulted in a spike or not.
            is_spike = self.voltage > 30

            # Recording the membrane potential.
            if record_values:
                self.vt_graph.record(time_i, 30 if is_spike else self.voltage)
                if is_spike:
                    self.spike_times.append(time_i)

            # Yield the membrane potential.
            if yield_values:
                yield self.voltage

            # Reset the voltage values if there has been a spike.
            if is_spike:
                self.calm_down()

        # A default yield if the function is not used as a generator.
        yield

    def apply_input_voltage(self, time_step, input_voltage):
        """
        This method calculates the membrane potential based on the given input voltage.
        This method makes use of the Izhikevich's Spiking Neuron equation to calculate the Na and K voltages.
        :param time_step: Float: The value of the time interval between one mini-step.
        :param input_voltage: Float: Input voltage to be supplied.
        :return: None. This only sets the instance members.
        """
        # Izhikevich's equations for spiking neuron.
        self.voltage += time_step * (
                0.04 * self.voltage ** 2 + 5 * self.voltage + 140 - self.recovery + input_voltage)
        self.recovery += time_step * self.a * (self.b * self.voltage - self.recovery)

    def calm_down(self):
        # Refractory period
        self.voltage = self.c
        self.recovery += self.d

    def get_spiking_rate(self, time_span=1000, percentage_from_last=80):
        """
        Calculates the spiking rate of this neuron and returns it.
        :param time_span: Int: The total time for which the stimulation was conducted.
        :param percentage_from_last: Float: The percentage of time steps from the last that will be considered.
        :return: Float: The spike rate of the neuron.
        """
        discarded_steps = int((1 - percentage_from_last / 100) * time_span)
        i = 0
        while len(self.spike_times) > i and self.spike_times[i] < discarded_steps:
            i += 1
        required_spikes = self.spike_times[i:]
        return len(required_spikes) / (time_span - discarded_steps)


class Grapher:
    """
    The whole purpose of this class is just to create some subplots and store the x and y data.
    """

    def __init__(self):
        self.x = []
        self.y = []

    def record(self, x_value, y_value):
        self.x.append(x_value)
        self.y.append(y_value)

    def plot(self, axis, title='Title', xlabel="x", ylabel="y", ylim=None, percentage_from_last=100):
        starting_index = int((1 - percentage_from_last / 100) * len(self.x))
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        if ylim:
            axis.set_ylim(ylim)
        axis.set_title(title)
        axis.plot(self.x[starting_index:], self.y[starting_index:])

    def clear_data(self):
        self.__init__()

    @staticmethod
    def create_figure(num_rows, num_columns, figure_number, figsize=(16, 10)):
        return plt.subplots(num_rows, num_columns, constrained_layout=True, num=figure_number, figsize=figsize, dpi=80)


def regular_spiking():
    """
    This is the demo run that the given matlab code would run.
    """
    neuron = SpikingNeuron()
    next(neuron.stimulate_input(lambda time: 1 if time > 50 else 0, time_span=1000, time_step=0.25))
    fig, axs = neuron.vt_graph.create_figure(num_rows=1, num_columns=1, figure_number=1)
    neuron.vt_graph.plot(axs)
    plt.show()


def simulate_constant_inputs_range(min_input, max_input, voltage_step, time_span=1000, time_step=0.25, num_rows=1,
                                   num_cols=1, save_figures=False):
    """
    This is the core part of the problem 1.
    Given a range of input values and their step difference, this function will iterate for each voltage value
    and supply that voltage to the neuron, then plot the graph for that input voltage.
    Once it has plotted graphs for all the input values, then this function will draw the graph of
    spiking rate over the input voltages supplied.
    :param min_input: Float: The starting value of the input voltage.
    :param max_input: Float: The ending value of the input voltage. This will be included in the stimulation.
    :param voltage_step: Float: The step value for each input iteration.
    :param time_span: Int: The time span for each input voltage.
    :param time_step: Float: The time step for each time-mini-step.
    :param num_rows: Int: Number of rows of subplots desired.
    :param num_cols: Int: Number of columns of subplots desired.
    :param save_figures: Boolean: It will save the graphs in figures directory if enabled.
    :return: None.
    """
    # Get the number of figures required for showing all the graphs with different input voltages.
    num_figures = math.ceil((max_input - min_input) / voltage_step / num_rows / num_rows)

    rt_graph = Grapher()

    # Iteration for each figure with multiple subplots.
    for fig_i in range(num_figures):

        fig, axs = Grapher.create_figure(num_rows, num_cols, fig_i)
        axs = axs.flatten()
        fig.suptitle(f'Regular Spiking Neuron Figure {fig_i + 1}', fontsize=16)

        # Iteration for each subplot with a single input voltage.
        for i in range(num_cols * num_rows):

            # Calculating the input voltage from the range given.
            input_voltage = min_input + voltage_step * i + (fig_i * num_rows * num_cols * voltage_step)
            if input_voltage > max_input:
                break

            neuron = SpikingNeuron(neuron_type='regular')

            # This is the stimulation part that will generate the spikes.
            next(neuron.stimulate_input(lambda time: input_voltage, time_span=time_span, time_step=time_step))

            # Plot the data of membrane potentials at each time mini-step in the subplot.
            neuron.vt_graph.plot(axs[i], title=f"Input = {input_voltage}", xlabel="time step", ylabel="V_m",
                                 ylim=(-90, 40))

            # Calculate the spiking rate of that input voltage.
            spike_rate = neuron.get_spiking_rate()

            # Record the spike rate for future plotting.
            rt_graph.record(input_voltage, spike_rate)

            neuron.vt_graph.clear_data()
        if save_figures:
            plt.savefig(f'figures/vt_graph_{fig_i}')

    # Plot the graph of spiking rate vs input voltage.
    rt_graph.plot(rt_graph.create_figure(1, 1, 1, figsize=(4, 4))[1], title="Firing rate against input", xlabel="Input",
                  ylabel="Firing Rate")
    if save_figures:
        plt.savefig('figures/rt_graph')

    plt.show()


def simulate_constant_inputs(inputs, time_span=1000, time_step=0.25, save_figures=False):
    """
    This function stimulates the neuron with a given list of input voltages.
    :param inputs: List<Float>: This is the list of voltage values with which the neuron would be stimulated.
    :param time_span: Int: The complete time span for which the neuron will be stimulated.
    :param time_step: Float: The time mini step
    :param save_figures: Boolean: Specifies whether to save the plots or not.
    :return: None.
    """
    fig, axs = Grapher.create_figure(len(inputs), 1, 1, figsize=(4, 8))
    fig.suptitle(f'Regular Spiking Neuron Figure 1', fontsize=16)

    # Iterate for each input value supplied.
    for i in range(len(inputs)):
        input_voltage = inputs[i]
        neuron = SpikingNeuron()

        # Stimulate the neuron with the input value for the given time period.
        next(neuron.stimulate_input(lambda time: input_voltage, time_span=time_span, time_step=time_step))

        # Plot the graph with membrane potentials at each time mini-step.
        neuron.vt_graph.plot(axs[i], title=f"Input: {input_voltage}", xlabel="time step", ylabel="V_m", ylim=(-90, 40))

        neuron.vt_graph.clear_data()
    if save_figures:
        plt.savefig(f'figures/vt_graph_1')
    plt.show()


def main():
    regular_spiking()


if __name__ == "__main__":
    main()
