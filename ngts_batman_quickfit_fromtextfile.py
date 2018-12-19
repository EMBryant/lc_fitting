#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ngts_batman_quickfit.py
#  
#  26th November 2018
#  Edward Bryant <phrvdf@monju.astro.warwick.ac.uk>
#  
#  
#  

import argparse
import numpy as np
import batman as bm
from scipy.optimize import Bounds, minimize as mini
from scipy.stats import chisquare, sem
import matplotlib.pyplot as plt
import time

def ParseArgs():
    '''
    Function to parse command line arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('fn', type=str, help="Name of LC data file")
    parser.add_argument('per', type=float, help="Orbital period in days")
    parser.add_argument('epoc', type=float, help="Time of first transit centre")
    
    return parser.parse_args()
    
def lc_min(params, phase, flux, err):
	'''
	Function which calculates Chi2 value for a given set of input parameters.
	Function to be minimized to find best fit parameters
	'''
	
	#Define the system parameters for the batman LC model
	pm = bm.TransitParams()
	
	pm.t0 = 0.                #Time of transit centre
	pm.per = 1.               #Orbital period = 1 as phase folded
	pm.rp = params[0]         #Ratio of planet to stellar radius
	pm.a = params[1]          #Semi-major axis (units of stellar radius)
	pm.inc = params[2]        #Orbital Inclination [deg]
	pm.ecc = 0.               #Orbital eccentricity (fix circular orbits)
	pm.w = 90.                #Longitude of periastron [deg] (unimportant as circular orbits)
	pm.u = [0.3, 0.2]         #Stellar LD coefficients
	pm.limb_dark="quadratic"  #LD model
	
	#Initialize the batman LC model and compute model LC
	m = bm.TransitModel(pm, phase)
	f_model = m.light_curve(pm)
	fit_vals = np.sqrt((flux - f_model)**2 / err**2)
	fit_val = np.sum(fit_vals) / (len(fit_vals) - 4)
	return fit_val

def lc_bin(time, flux, bin_width):
	'''
	Function to bin the data into bins of a given width. time and bin_width 
	must have the same units
	'''
	
	edges = np.arange(np.min(time), np.max(time), bin_width)
	dig = np.digitize(time, edges)
	time_binned = (edges[1:] + edges[:-1]) / 2
	flux_binned = np.array([np.nan if len(flux[dig == i]) == 0 else flux[dig == i].mean() for i in range(1, len(edges))])
	err_binned = np.array([np.nan if len(flux[dig == i]) == 0 else sem(flux[dig == i]) for i in range(1, len(edges))])
	time_bin = time_binned[~np.isnan(err_binned)]
	err_bin = err_binned[~np.isnan(err_binned)]
	flux_bin = flux_binned[~np.isnan(err_binned)]	
	
	return time_bin, flux_bin, err_bin	

def lc_fit(fn, period, t0):
    '''
    Function to fit a batman model to an input lc datafile to find the best 
    fit system parameters
    '''
    #Load time and flux data for the object
    DATA = np.loadtxt(fn)
    time, flux, err = DATA[:, 0], DATA[:, 3], DATA[:, 4]
    
    phase = ((time - t0)/period)%1  #Convert time values in to phases
    phase = np.array([p-1 if p > 0.8 else p for p in phase], dtype=float)
    
    p_fit = phase[phase < 0.2]  #Crop phase and flux arrays to only contain values
    f_fit = flux[phase < 0.2]   #in range (-0.2 ,  0.2)
    e_fit = err[phase < 0.2]
    
    params = [0.05, 10., 90.]   #Input parameters [rp, a, inc] for minimization
    bnds = Bounds(([0., 0., 60.]), ([1., 100., 90.])) #Parameter upper, lower bounds
    
    res = mini(lc_min, params, args=(p_fit, f_fit, e_fit), bounds=bnds) #perform minimization
    
    rp_best, a_best, inc_best = res.x[0], res.x[1], res.x[2] #Best fit parameters
    print('Best fit parameters: rp={:.4f}; a={:.4f}; inc={:.4f}'.format(rp_best, a_best, inc_best))
    print('Minimization success: {};  chi2={:.4f}'.format(res.success, res.fun))
    
    #Produce a best fit model using the minimization results
    pm_best = bm.TransitParams()
    
    pm_best.t0 = 0.                #Time of transit centre
    pm_best.per = 1.               #Orbital period = 1 as phase folded
    pm_best.rp = rp_best           #Ratio of planet to stellar radius
    pm_best.a = a_best             #Semi-major axis (units of stellar radius)
    pm_best.inc = inc_best         #Orbital Inclination [deg]
    pm_best.ecc = 0.               #Orbital eccentricity (fix circular orbits)
    pm_best.w = 90.                #Longitude of periastron [deg] (unimportant as circular orbits)
    pm_best.u = [0.3, 0.2]        #Stellar LD coefficients
    pm_best.limb_dark="quadratic"  #LD model
    
    p_best = np.linspace(-0.2, 0.2, 10000)
    m_best = bm.TransitModel(pm_best, p_best)
    f_best = m_best.light_curve(pm_best)
    
    p1 = p_best[np.where(f_best < 1)[0][0]]    #Phase of first contact
    p4 = p_best[np.where(f_best < 1)[0][-1]]   #Phase of final contact
    
    t_dur = (p4 - p1) * period *24             #Transit duration [hours]
    t_depth = (1 - f_best.min()) * 100         #Transit depth [percent]
    
    #Produce binned data set for plotting
    bw = 10 / (1440*period)                    #Bin width - 10 mins in units of phase
    p_bin, f_bin, e_bin = lc_bin(p_fit, f_fit, bw)
    
    #Produce plot of data and best fit model LC
    plt.figure()
    
    plt.plot(p_fit, f_fit, marker='o', color='gray', linestyle='none', markersize=1)
    plt.plot(p_bin, f_bin, 'ro', markersize=3)
    plt.plot(p_best, f_best, 'g--')
    
    plt.xlabel('Phase')
    plt.ylabel('Flux')
    plt.title('Depth: {:.4f}%;  Duration: {:4f} hours;  (Rp/Rs): {:.4f}'.format(t_depth, t_dur, rp_best))
    
# 	plt.savefig("Plot save location")
    plt.show()
    
    return p_best, f_best, p_fit, f_fit, rp_best, a_best, inc_best, t_dur, t_depth
 	

if __name__ == '__main__':
    
    args = ParseArgs()
    
    time0 = time.time()
    p_best, f_best, p_fit, f_fit, rp_best, a_best, inc_best, t_dur, t_depth = lc_fit(args.fn, args.per, args.epoc)
    time1 = time.time()
    print("Time taken: {:.4f} s".format(time1-time0))
