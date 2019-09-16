from Practice.gradient_descent.spiking_nurons.neuron import SpikingNeuron, Grapher
import matplotlib.pyplot as plt

figure_number = 1


def accumulate_inputs(time, neuron_a, neuron_b, weight):
    time_index = int(time / 0.25)
    voltage_a = neuron_a.vt_graph.y[time_index]
    voltage_b = neuron_b.vt_graph.y[time_index]

    input_a = weight if voltage_a >= 30 else 0
    input_b = weight if voltage_b >= 30 else 0

    return input_a + input_b


def plot_neurons(neuron_a, neuron_b, neuron_c, weight):
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
    fig, axs = Grapher.create_figure(1, 1, 10000, (4, 4))
    axs.set_xlabel("Weights")
    axs.set_ylabel("Spike Rate")
    axs.set_xticks(range(50, 151, 25))
    axs.set_title("Spike rate vs Weights")
    axs.plot(weights, spike_rate)


def main():
    weights = [weight for weight in range(50, 151, 10)]
    spike_rates = []
    for weight in weights:
        neuron_a = SpikingNeuron()
        a_gen = neuron_a.simulate_input(lambda time: 5, 1000, 0.25, yield_values=True)

        neuron_b = SpikingNeuron()
        b_gen = neuron_b.simulate_input(lambda time: 15, 1000, 0.25, yield_values=True)

        neuron_c = SpikingNeuron()
        c_gen = neuron_c.simulate_input(lambda time: accumulate_inputs(time, neuron_a, neuron_b, weight), 1000, 0.25,
                                        yield_values=True)

        for i in range(4001):
            next(a_gen)
            next(b_gen)
            next(c_gen)

        plot_neurons(neuron_a, neuron_b, neuron_c, weight)
        # plt.savefig(f"figures/simple_network_{weight}")
        spike_rates.append(neuron_c.get_spiking_rate())

    plot_spiking_rate(spike_rates, weights)
    # plt.savefig("figures/rate_weight")

    plt.show()


if __name__ == "__main__":
    main()
