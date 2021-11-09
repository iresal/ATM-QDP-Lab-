"""
QDP Lab data

"""
import os
import numpy as np
from numpy.core.fromnumeric import size
from numpy.lib.npyio import save 
import scipy.io as sio
import scipy.optimize
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import matplotlib as mpl
import matplotlib.pyplot as plt 
import pandas as pd
from sympy import *

# define the true objective function
def objective(x, a, b, c,d):
	return a * x + b * x**2 +d
    #return  1/(a+(x*b**2/c*(x-0)))

def init_fig_params():
    ####### Figure parameters 
    mpl.rc('axes',edgecolor='k')

    plt.rcParams['font.family'] = 'Arial'
    mpl.rcParams.update({'font.size': 8})

def freq_swept (namefile):
    freq = np.loadtxt(namefile)[:, 0]
    real = np.loadtxt(namefile)[:, 1]*-1
    im = np.loadtxt(namefile)[:, 2]
    phase_deg = np.loadtxt(namefile)[:, 3]
    sio.savemat(data_path+'datafile2021-10-13freq_sweep_4.mat',{'freq':freq,'real':real,'im':im,'phase_deg':phase_deg})
    
    ### Real part 
    f_swept_figure = plt.figure(figsize=(4,3))
    ax = f_swept_figure.add_subplot(111)
    ax.plot(freq,real*1e3)
    ax.plot(73.24, 0.48835,'x')
    ax.set_xscale('log')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Real [mV]')
    #ax.set_ylim([-0.6,-0.46])
    plt.axvline(73.24, 0, 1, color = 'k',ls=':',lw=1) 
   
    xy = [75, 0.48]
                                         # <--
    ax.annotate('73.24 Hz, 0.48 mV', xy=xy, textcoords='data') 

    plt.tight_layout()
    plt.savefig(figures_path+'freq_swept'+'.pdf', format = 'pdf', dpi=300)
    f_swept_figure.show()
    ### Imag part

    f_swept_figure_IM = plt.figure(figsize=(4,3))
    ax = f_swept_figure_IM.add_subplot(111)
    ax.plot(freq,im*1e3)
    ax.plot(73.24, 0.00732,'x')
    ax.set_xscale('log')
    ax.set_xlabel('Freq [Hz]')
    ax.set_ylabel('IM [mV]')
    ax.set_ylim([-0.1,0.12])
    plt.axvline(73.24, 0, 1, color = 'k',ls=':',lw=1) 
   
    xy = [78, -0.005]
                                         # <--
    ax.annotate('73.24 Hz, 0.02 mV', xy=xy, textcoords='data') 

    
    plt.tight_layout()
    plt.savefig(figures_path+'freq_swept_IM'+'.pdf', format = 'pdf', dpi=300)
    f_swept_figure_IM.show()
    
        


if __name__ == "__main__": 

    init_fig_params()
    ## Path of the files 
    data_path = '..Data\\'
    figures_path = '...\\Figures\\'
    
    file_name = 'datafile2021-10-13freq_sweep_4.txt'
    freq_swept (data_path+file_name)


    input()
