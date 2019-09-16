from Practice.gradient_descent.spiking_nurons import neuron


def main():
    neuron.simulate_constant_inputs_range(voltage_step=0.25, min_input=0, max_input=20, num_rows=4, num_cols=5,
                                          time_span=1000, time_step=0.1, save_figures=True)

    neuron.simulate_constant_inputs(inputs=[1, 5, 10, 15, 20], save_figures=True)


if __name__ == "__main__":
    main()
