"""
Author: Akhilez.com
Date: 9/13/19
This is the solution for the problem 1.
"""

import neuron


def main():
    # This method call is for observing the spikes of the neuron at input voltages ranging from 0 to 20
    neuron.simulate_constant_inputs_range(voltage_step=0.25, min_input=0, max_input=20, num_rows=4, num_cols=5,
                                          time_span=1000, time_step=0.1, save_figures=True)

    # This method call is for observing the spikes of the neuron at input voltages [1, 5, 10. 15, 20]
    neuron.simulate_constant_inputs(inputs=[1, 5, 10, 15, 20], save_figures=True)


if __name__ == "__main__":
    main()
