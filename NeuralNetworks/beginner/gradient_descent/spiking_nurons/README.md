## Spiking Neurons!
Make sure that you run with python 3.6 or greater. 

To run problem 1:
```
python rate_code.py
```
This will generate the graphs for  observations of 81 inputs and the 1-20 inputs along with the spiking rate graph.
Please note that plotting both voltage graph and spiking rate graph at one will cause spiking rate graph to look weird. That is how matplotlib works, unfortunately.
To get a better spiking rate graph, you have to stop the voltage graphs from plotting. That can be done by commenting the plot method call.

To run problem 2:
```bash
python simple_network.py
```
This will generate 11 graphs of voltage vs time and one spiking rate cs weights. 