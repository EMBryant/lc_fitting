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
from scipy.stats import chisquare, sem
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, Parameter, report_fit
from time import time as TIME

def ParseArgs():
    '''
    Function to parse command line arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('fn', type=str, help="Name of LC data file")
    parser.add_argument('objid', type=str, help="NGTS object id")
    parser.add_argument('--fit_t0', action='store_true', help="Add this to allow t0 to vary in the fitting")
    
    return parser.parse_args()
    
def lc_min(params, phase, flux, err, fit_t0):
	'''
	Function which calculates Chi2 value for a given set of input parameters.
	Function to be minimized to find best fit parameters
	'''
	
	#Define the system parameters for the batman LC model
	pm = bm.TransitParams()
	
	if fit_t0:
		pm.t0 = params['t0'].value
	else:
		pm.t0 = 0.                   #Time of transit centre
	pm.per = 1.                  #Orbital period = 1 as phase folded
	pm.rp = params['rp'].value   #Ratio of planet to stellar radius
	pm.a = params['a'].value             #Semi-major axis (units of stellar radius)
	pm.inc = params['inc'] .value          #Orbital Inclination [deg]
	pm.ecc = 0.                  #Orbital eccentricity (fix circular orbits)
	pm.w = 90.                   #Longitude of periastron [deg] (unimportant as circular orbits)
	pm.u = [0.1, 0.3]            #Stellar LD coefficients
	pm.limb_dark="quadratic"     #LD model
	
	#Initialize the batman LC model and compute model LC
	m = bm.TransitModel(pm, phase)
	f_model = m.light_curve(pm)
	residuals = (flux - f_model)**2/err**2
	return residuals

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

def lc_fit(fn, period, epoc, objid, fit_t0=False):
    '''
    Function to fit a batman model to an input lc datafile to find the best 
    fit system parameters
    '''

    time0 = TIME()
    #Load time and flux data for the object
    DATA = np.loadtxt(fn)
    time, flux, err = DATA[:, 0], DATA[:, 3], DATA[:, 4]
    
    phase = ((time - epoc)/period)%1  #Convert time values in to phases
    phase = np.array([p-1 if p > 0.8 else p for p in phase], dtype=float)
    
    p_fit = phase[phase < 0.2]  #Crop phase and flux arrays to only contain values
    f_fit = flux[phase < 0.2]   #in range (-0.2 ,  0.2)
    e_fit = err[phase < 0.2]
    
    params=Parameters()         #Parameter instance to hold fit parameters
    params.add('rp', value=0.05, min=0., max=1.)    #Planet:Star radius ratio
    params.add('a', value=10., min=0., max=100.)    #Semi-major axis
    params.add('inc', value=89., min=60., max=90.)  #Orbital inclination
    if fit_t0: 
        params.add('t0', value=0.0, min=-0.1, max=0.1) #Transit centre time
    
    res = minimize(lc_min, params, args=(p_fit, f_fit, e_fit, fit_t0), method='leastsq') #perform minimization
    chi2 = np.sum(res.residual) / res.nfree
    rp_best, a_best, inc_best = res.params['rp'].value, res.params['a'].value, res.params['inc'].value
    if fit_t0: 
        t0_best = res.params['t0'].value
        print('Best fit parameters: rp={:.6f}; a={:.6f}; inc={:.6f}; t0={:.6f}'.format(rp_best, a_best, inc_best, t0_best))
    else:	
        print('Best fit parameters: rp={:.6f}; a={:.6f}; inc={:.6f}'.format(rp_best, a_best, inc_best))
    
    print('Minimization result: {}: {}; chi2={:.4f}'.format(res.success, res.message, chi2))
    
    #Produce a best fit model using the minimization results
    pm_best = bm.TransitParams()
    
    if fit_t0: pm_best.t0 = t0_best
    else: pm_best.t0 = 0.                #Time of transit centre
    
    pm_best.per = 1.               #Orbital period = 1 as phase folded
    pm_best.rp = rp_best           #Ratio of planet to stellar radius
    pm_best.a = a_best             #Semi-major axis (units of stellar radius)
    pm_best.inc = inc_best         #Orbital Inclination [deg]
    pm_best.ecc = 0.               #Orbital eccentricity (fix circular orbits)
    pm_best.w = 90.                #Longitude of periastron [deg] (unimportant as circular orbits)
    pm_best.u = [0.1, 0.3]        #Stellar LD coefficients
    pm_best.limb_dark="quadratic"  #LD model
    
    p_best = np.linspace(-0.2, 0.2, 10000)     #Produce a model LC using 
    m_best = bm.TransitModel(pm_best, p_best)  #the output best fit parameters
    f_best = m_best.light_curve(pm_best)
    
    p1 = p_best[np.where(f_best < 1)[0][0]]    #Phase of first contact
    p4 = p_best[np.where(f_best < 1)[0][-1]]   #Phase of final contact
    
    t_dur = (p4 - p1) * period *24             #Transit duration [hours]
    t_depth = (1 - f_best.min()) * 100         #Transit depth [percent]
    
    #Produce binned data set for plotting
    bw = 10 / (1440*period)                    #Bin width - 10 mins in units of phase
    p_bin, f_bin, e_bin = lc_bin(p_fit, f_fit, bw)
    
    #Produce plot of data and best fit model LC
    plt.figure(figsize=(9, 7.5))
    
    plt.plot(p_fit, f_fit, marker='o', color='gray', linestyle='none', markersize=1)
    plt.plot(p_bin, f_bin, 'ro', markersize=5)
    plt.plot(p_best, f_best, 'g--', linewidth=2)
    
    plt.xlabel('Phase')
    plt.ylabel('Flux')
    plt.title('Depth: {:.4f}%;  Duration: {:4f} hours;  (Rp/Rs): {:.4f};  chi2: {:.4f}'.format(t_depth, t_dur, rp_best, chi2))
    plt.xlim((-3*p4, 3*p4))
    plt.ylim((f_bin.min()-t_depth/200, 1+t_depth/200))
    
#    plt.savefig('/home/astro/phrvdf/NGTS_fitting/quickfit_plots/NOI_{}_lcfit.png'.format(objid))
    time1 = TIME()
    print("Time taken: {:.4f} s".format(time1-time0))

    plt.show()
    if fit_t0:
        return p_best, f_best, p_fit, f_fit, e_fit, rp_best, a_best, inc_best, t0_best, t_dur, t_depth, res
    else:	
        return p_best, f_best, p_fit, f_fit, e_fit, rp_best, a_best, inc_best, t_dur, t_depth, res

if __name__ == '__main__':
    
    args = ParseArgs()
    
    df = pandas.read_csv('/home/astrp/phrvdf/NOI_TESS_crossovers/sector1/ads/noi_transit_times.csv', index_col='noi')
    epoc = df.loc[args.objid, 'epoc']
    per = df.loc[args.objid, 'per']
    
    if args.fit_t0:
        p_best, f_best, p_fit, f_fit, e_fit, rp_best, a_best, inc_best, t0_best, t_dur, t_depth, res = lc_fit(args.fn, per, epoc, args.objid, fit_t0=True)
    else:
        p_best, f_best, p_fit, f_fit, e_fit, rp_best, a_best, inc_best, t_dur, t_depth, res = lc_fit(args.fn, per, epoc, args.objid, fit_t0=False)
    
