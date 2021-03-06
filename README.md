# ATM-QDP-Lab-
This repository contains the data recorded in the Quantum Device Physics within the Advanced Measurement techniques course. Measurements were performed by Rasmus Larsson and Israel Rebolledo. 

The files are stored in the following folders: 

- Data: 
    a)  time_sweeps.mat - Data corresponding to the device cooling measurements.
    b)  datafile2021-10-13freq_sweep_4.txt  - Measurement of the freqency characterization of the lock-in carrier using a 10kOhm resistor \n
    c)  pinchoffdata.mat - Data corresponding to the device characterizations. 
    - Raw Data. Folder with all text files recorded during the lab session. The MAT files contained the data used for the post-processing of the measurements 
    
- Measurement Jupyter Notebooks:
    a) AMT_2021_part1-03.ipynb - Notebook used to record the temperature measurements 
    b) AMT_2021_part2-03.ipynb - Notebook used to record the device characterization measurements. 
    
 - Post processing scripts: 
    a) Plotting_and_postprocessing.m - This Matlab script plot the data and fit the measurements of the temperature cooling and pinch off traces. 
    b) Plotting_frequency_swept.py - This python script is used to plot the frequency swept of the carrier reference signal of the lock-in amplifier. 
