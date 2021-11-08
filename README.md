# ATM-QDP-Lab-
This repository contains the data recorded in the Quantum Device Physics within the Advanced Measurement techniques course. 

The files are stored in the following folders: 

- Data: 
    a)  time_sweeps.mat - Data corresponding to the decive cooling measurements. The mat file already includes all text files recorded in the lab session (Raw Data folder)
    b)  datafile2021-10-13freq_sweep_4.txt  - Measurement of the freqency characterization of the lock-in carrier using a 10kOhm resistor 
    c)  pinchoffdata.mat - Data corresponding to the device characterizations. The mat file already includes all text files recorded in the lab session (Raw Data folder)
    
- Measurement Jupyter Notebooks:
    a) AMT_2021_part1-03.ipynb - Notebook used to record the temperature measurements 
    b) AMT_2021_part2-03.ipynb - Notebook used to record the device characterization measurements. 
    
 - Post processing scripts: 
    a) Plotting_and_postprocessing.m - This Matlab script plot the data and fit the measurements of the temperature cooling and pinch off traces. 
    b) Plotting_frequency_swept.py - This python script is used to plot the frequency swept of the carrier reference signal of the lock-in amplifier. 
