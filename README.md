Folder 1 contains 3 files demonstrating an example simulation of the spiking neuron model.

- start.py: example run of network III with no suppression
- SpikingNeuronModel.py: library for the simulation
- network_III.txt.zip: zipped file for coupling strength matrix of network III

Python libraries required:

- numpy
- matplotlib
- tqdm

To start the simulation, run start.py in terminal:
'''
python start.py
'''

After the simulation, the data and sample plots will be stored in 'out-plot/' and 'out-cont/' respectively. 'out-plot/' contains raster plots and time series plots. 'out-cont/' contains the time series of the membrane potential v and recovery variable u, the timestamp of the spikes and other files for continue the simulation marked with (cont).

The raw data are all in Numpy .npy binary file format:
- out-spike: timestamp of the spikes
- out-voltage: time series of the membrane potential v
- out-recovery: time series of the recovery variable u

Parameters in start.py:
- totTime: total simulation time in ms
- timeStep: intergration time step in ms
- plotStep: time steps between each plotting of intermediate time series files
- network: coupling strength matrix of the network
- loadFromPrev: True if continuing the simulation

