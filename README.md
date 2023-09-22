# Coupling Analysis of a two transmon experiment

## Summary
This code is part of my master thesis on the simulation of coupled transmons circuit in order to design a experiment of Wigner tomography. It contains 3 circuits, a direct coupling of two transmon, a transmon coupled to a resonator and a two transmon system coupled via a resonator. There is also a notebook for the simulation of the energy level of the transmon qubit to test the method. The parameter for each simulation have be set to match the parameter of a paper that is ' Eric M Kessler. “Generalized Schrieffer-Wolff formalism for dissipative systems”'.

## Requierements

The code runs on python and requiries the following packages:
- numpy
- scipy
- qutip
- matplotlib
- tqdm

Please be sure to have these packages isntalled on your python distribution before executing the code.
## Code structure
For each circuit correspond a jupyter notebook and a python class. All the code for the simulation and the method of nodes is in the python calss it should no be modified unless with knoledge of the code and the method. So on the main files are each jupyter notebook for each circuits. They are composed of many simulations and plots and the circuit and hamiltonian studied are reminded before each section.

## How to use it
Open the jupyter notebook for each circuit you want to simulate. Launch the first cell for the import at the beginning of the use of the notebook. Then Launch each cells for each simulations and plot. you can tweak the parameters at the beggining of each cells to modify the simulation parameters.

## References

``` 
Eric M Kessler. “Generalized Schrieffer-Wolff formalism for dissipative systems”.
Denis Vion. “Josephson quantum bits based on a cooper pair box”. 
Alexandre Blais et al. “Circuit quantum electrodynamics”.


