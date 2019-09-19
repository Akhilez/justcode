"""
Author: Akhilez.com
Date: 9/13/19
Description: This file contains the solution to the second problem.
The challenge is to create a simple network of 3 Spiking Neurons and studying their spikes.
The main function will plot the voltage vs time graphs for neuron A, B and C as well as the spiking rate vs weight graph
"""

from neuron import SpikingNeuron, Grapher
import matplotlib.pyplot as plt

figure_number = 1


def accumulate_inputs(time, neuron_a, neuron_b, weight):
    """
    This function will simply aggregate the inputs after scaling them with their weights.
    Here, the weight for all inputs is the same ie weight param.
    :param time: Float: Time at which the input is being given.
    :param neuron_a: SpikingNeuron object
    :param neuron_b: SpikingNeuron object
    :param weight: Float: The weight for the inputs.
    :return: The aggregate of all the inputs.
    """
    time_index = int(time / 0.25)
    voltage_a = neuron_a.vt_graph.y[time_index]
    voltage_b = neuron_b.vt_graph.y[time_index]

    input_a = weight if voltage_a >= 30 else 0
    input_b = weight if voltage_b >= 30 else 0

    return input_a + input_b


def plot_neurons(neuron_a, neuron_b, neuron_c, weight):
    """
    This function simply plots the graph of membrane potential vs time for each neuron.
    :param neuron_a: SpikingNeuron
    :param neuron_b: SpikingNeuron
    :param neuron_c: SpikingNeuron
    :param weight: Float: Weight for each input.
    :return: None.
    """
    # Figure number is used to distinguish one figure from another.
    global figure_number
    fig, axs = Grapher.create_figure(num_rows=3, num_columns=1, figure_number=figure_number, figsize=(4, 6))
    figure_number += 1
    fig.suptitle(f"Simple network of 3 neurons. Weight: {weight}")

    neuron_a.vt_graph.plot(axis=axs[0], xlabel="time step", ylabel="V_m", ylim=(-90, 40), title="Neuron A. Input: 5",
                           percentage_from_last=20)
    neuron_b.vt_graph.plot(axis=axs[1], xlabel="time step", ylabel="V_m", ylim=(-90, 40), title="Neuron B. Input: 15",
                           percentage_from_last=20)
    neuron_c.vt_graph.plot(axis=axs[2], xlabel="time step", ylabel="V_m", ylim=(-90, 40),
                           title=f"Neuron C. Weight: {weight}", percentage_from_last=20)


def plot_spiking_rate(spike_rate, weights):
    """
    This function plots the graph of spiking rate vs weights.
    :param spike_rate: List<Float>: Spiking rate for each weight.
    :param weights: List<Float>: Weights.
    :return: None.
    """
    fig, axs = Grapher.create_figure(1, 1, 10000, (4, 4))
    axs.set_xlabel("Weights")
    axs.set_ylabel("Spike Rate")
    axs.set_xticks(range(50, 151, 25))
    axs.set_title("Spike rate vs Weights")
    axs.plot(weights, spike_rate)


def main():
    """
    This function will create a network of 3 spiking neurons.
    For the weights ranging from 50 to 150 in a step of 10, the neurons are stimulated for 1000 ms.
    The neuron A gets a constant input voltage of 5 and neuron B gets 15.
    The outputs from the neurons A and B are considered as binary. As in 1 if there is a spike. 0 otherwise.
    The plots of voltage vs time and spiking rate vs weights will be plotted.
    :return: None
    """
    weights = [weight for weight in range(50, 151, 10)]
    spike_rates = []

    # Iterate for each weight.
    for weight in weights:

        # Get the iterator for neuron A
        neuron_a = SpikingNeuron()
        a_gen = neuron_a.stimulate_input(lambda time: 5, 1000, 0.25, yield_values=True)

        # Get the iterator for neuron B
        neuron_b = SpikingNeuron()
        b_gen = neuron_b.stimulate_input(lambda time: 15, 1000, 0.25, yield_values=True)

        # Get the iterator for neuron C
        # Here the input function will be the aggregator function.
        neuron_c = SpikingNeuron()
        c_gen = neuron_c.stimulate_input(lambda time: accumulate_inputs(time, neuron_a, neuron_b, weight), 1000, 0.25,
                                         yield_values=True)

        # Run the stimulation for 1000 ms with 0.25 mini-step, ie calculate voltage at every 0.25 ms
        for i in range(4001):
            next(a_gen)
            next(b_gen)
            next(c_gen)

        # Plot the voltage vs time graphs.
        plot_neurons(neuron_a, neuron_b, neuron_c, weight)
        # plt.savefig(f"figures/simple_network_{weight}")

        # Record the spiking rate for the current weight.
        spike_rates.append(neuron_c.get_spiking_rate())

    # Plot the graph of spiking rate vs weights for all the weights.
    plot_spiking_rate(spike_rates, weights)
    # plt.savefig("figures/rate_weight")

    plt.show()


if __name__ == "__main__":
    main()
