# ABM_PanelWaves

Source code for the simulation of respondents in a panel survey. The simulation consists of a python script and two sets of input data: The data on respondents (NAME) and the data on media input (NAME). The simulation requires all three files in the same folder. Other files are not required and the script uses only basic libraries (time, math, random, os).

The data was collected in a three-wave panel survey and accompanying content analysis in the run-up of a referendum in Switzerland in 2006. The issue of the referendum was a more restrictive legislation concerning asylum seekers (pro: instal more restrictive laws / con: prevent these laws).

The simulation used all respondents as agents with their respective attitudes and their location in Switzerland. At each point in time, each respondent consults the media outlets they have indicated to use and perceives the opinion of all other agents. From the bias in the media and their environment, each agent slightly adjusts their opinion.
Opinion change is dependent on overall media and social impact and on individual susceptibility to influences and attitude strength.


Attached to the project you find a PDF (SimulatingGaps.pdf) containing a presentation given at the 2018 annual convention of the ICA. It contains additional information on the studies, the contents of the program and results from the simulations.
