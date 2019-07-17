# ABM_PanelWaves

Source code for the simulation of respondents in a panel survey. The simulation consists of a python script and two sets of input data: The data on respondents (Survey_Data.dat) and the data on media input (Content_Data.dat). The simulation requires all three files in the same folder. Other files are not required and the script uses only basic libraries (time, math, random, os).

The data was collected in a three-wave panel survey and accompanying content analysis in the run-up of a referendum in Switzerland in 2006. The issue of the referendum was a more restrictive legislation concerning asylum seekers (pro: install more restrictive laws / con: prevent these laws).

The simulation used all respondents as agents with their respective attitudes and their location in Switzerland. At each point in time, each respondent consults the media outlets they have indicated to use and perceives the opinion of all other agents. From the bias in the media and their environment, each agent slightly adjusts their opinion.
Opinion change is dependent on overall media and social impact and on individual susceptibility to influences and attitude strength.
A genetic algorithm is used to find the parameter values that inform the ABM to behave as the respondents in the panel survey behaved over three waves. The algorithm uses the agreement of the final state of the simulation with the following wave of the survey as fitness measure to find the optimal parameters.


The repository contains several folders, each complete with one implementation of the genetic algorithm and an R-script that visualizes the results obtained from the simulation. The difference between the implementations just lies in parameter values specified at the beginning of the program and some changes in the R-scripts.
 * **Simulation_Naive_W1**: This implementation starts with the null hypothesis of there not being any influence at all and updates this assumption. It is quite slow to converge on optimal values for all parameters. It eventually finds the optimal parameter values for the first panel interval between the first and second panel wave.
 * **Simulation_BS_W1**: This implementation starts with the optimal values for the first panel interval found by naive parameter search and updates the values to a sub-sample of the original sample over 70 generations before resetting the initial values and drawing another sub-sample. Repeating this routine 50 times, the standard error of parameter values become apparent.
 * **Simulation_BS_W2**: This implementation does the same thing as BS_W1 but for the second panel interval between the second and third panel wave.
* **Simulation_Integ_W1**: This implementation starts with the null hypothesis of there not being any effects and updates it, similar to the Naive_W1 implementation. However, each generation of the evolutionary algorithm uses a different sub-sample of the original sample to prevent overfitting to one sample. This implementation is considerably faster than Naive_W1 and bootstrapping becomes obsolete.
* **Simulation_Integ_W2**: This implementation does the same as Integ_W1 but for the second panel interval.
* **Visualization**: This implementation does not use the genetic algorithm but just does one run of the simulation with specified parameters. In the end of the simulation, the final state of the opinion climate is visualized in pop-up containing a TkInter canvas. This script also allows for the storage of individual agent carreers.


Attached to the project you find a PDF (SimulatingGaps.pdf) containing a presentation given at the 2018 annual convention of the ICA. It contains additional information on the studies, the contents of the program and results from the simulations.
