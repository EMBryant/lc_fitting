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
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, Parameter, report_fit
from time import time as TIME
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

def ParseArgs():
    '''
    Function to parse command line arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('fn', type=str, help="Name of LC data file")
    parser.add_argument('objid', type=str, help="NGTS object id")
    parser.add_argument('--epoch', type=float, help="Transit Epoch")
    parser.add_argument('--per', type=float, help="Orbital period")
    parser.add_argument('--dur', type=float, help="Transit Duration")
    parser.add_argument('--ecc', type=float, default=0., help="Orbital Eccentricity")
    parser.add_argument('--w', type=float, default=90., help="Argument of periastron")
    parser.add_argument('--u1', type=float, default=0.3, help="LD Coeff 1")
    parser.add_argument('--u2', type=float, default=0.1, help="LD Coeff 2")
    parser.add_argument('--ld', type=str, default="quadratic", help="LD formula")
    
    return parser.parse_args()
    
def lc_min(params, time, flux, err, spl):
	'''
	Function which calculates Chi2 value for a given set of input parameters.
	Function to be minimized to find best fit parameters

	Uses a spline to model any Out-of-Transit variability
	'''
	
	phase = ((time - (params['t0'].value))/params['per'].value)%1

    #Produce binned data set for caclulating the spline
	variability_model = spl(phase)
	
	#Define the system parameters for the batman LC model
	pm = bm.TransitParams()
	
	pm.t0 = 0                    #Time of transit centre
	pm.per = 1.                  #Orbital period = 1 as phase folded
	pm.rp = params['rp'].value   #Ratio of planet to stellar radius
	pm.a = params['a'].value             #Semi-major axis (units of stellar radius)
	pm.inc = params['inc'] .value          #Orbital Inclination [deg]
	pm.ecc = args.ecc                  #Orbital eccentricity (fix circular orbits)
	pm.w = args.w                   #Longitude of periastron [deg] (unimportant as circular orbits)
	if not args.ld == "uniform": pm.u = [args.u1, args.u2]            #Stellar LD coefficients
	else: pm.u = []
	pm.limb_dark=args.ld     #LD model
	
	#Initialize the batman LC model and compute model LC
	m = bm.TransitModel(pm, phase)
	f_model = m.light_curve(pm)
	model_total = f_model*variability_model
	residuals = (flux - model_total)**2/err**2
	return residuals

def poly_min(params, phase, flux, err):
     
    poly_model = params['b'].value*phase**2 + params['c'].value*phase + params['d'].value
  
    residuals = (flux - poly_model)**2/err**2
    return residuals

def lc_bin(time, flux, err, bin_width):
	'''
	Function to bin the data into bins of a given width. time and bin_width 
	must have the same units
	'''
	
	edges = np.arange(np.min(time), np.max(time), bin_width)
	dig = np.digitize(time, edges)
	time_binned = (edges[1:] + edges[:-1]) / 2
	flux_binned = np.array([np.nan if len(flux[dig == i]) == 0 else flux[dig == i].mean() for i in range(1, len(edges))])
	err_binned = np.array([np.nan if len(err[dig == i]) == 0 else np.sqrt(np.sum(err[dig == i]**2))/len(err[
    dig == i]) for i in range(1, len(edges))])
	time_bin = time_binned[~np.isnan(err_binned)]
	err_bin = err_binned[~np.isnan(err_binned)]
	flux_bin = flux_binned[~np.isnan(err_binned)]	
	
	return time_bin, flux_bin, err_bin	

if __name__ == '__main__':
    
    args = ParseArgs()

    period = args.per
    epoch = args.epoch
    t_dur = args.dur
    objid = args.objid

    time0 = TIME()
    #Load time and flux data for the object
    DATA = np.loadtxt(args.fn)
    t, flux, err = DATA[:, 0] - 2450000, DATA[:, 3], DATA[:, 4]
    
    time = np.array(t, dtype=float)
        
    params=Parameters()         #Parameter instance to hold fit parameters

    #TransitModel Parameters
    params.add('rp', value=0.05, min=0., max=1.)    #Planet:Star radius ratio
    params.add('a', value=10., min=0., max=100.)    #Semi-major axis
    params.add('inc', value=89., min=60., max=90.)  #Orbital inclination
    params.add('t0', value=epoch, min=epoch-period*0.1, max=epoch+period*0.1) #Transit centre time
    params.add('per', value=period, min=period-1, max=period+1)

    params_poly=Parameters()
    #Polyfit Parameters
    params_poly.add('b', value=-1, min=-100, max=100)
    params_poly.add('c', value=-1, min=-100, max=100)
    params_poly.add('d', value=-1, min=-100, max=100)
    
    hw = (t_dur/2) * 1.1 / 24 / period
    
    p = ((time - epoch)/period)%1
    p = np.array([p-1 if p > 0.5 else p for p in p])
    
    p_before = p[p < -1*hw]
    f_before = flux[p < -1*hw]
    e_before = err[p < -1*hw]
    p_after = p[p > hw]
    f_after = flux[p > hw]
    e_after = err[p > hw]
    
    res_before = minimize(poly_min, params_poly, args=(p_before, f_before, e_before), method="leastsq")
    res_after = minimize(poly_min, params_poly, args=(p_after, f_after, e_after), method="leastsq")

    poly_model_before = res_before.params['b']*p_before**2 + res_before.params['c']*p_before + res_before.params['d']
    poly_model_after = res_after.params['b']*p_after**2 + res_after.params['c']*p_after + res_after.params['d']
    p_oot = np.append(p_before, p_after)
    poly_model_oot = np.append(poly_model_before, poly_model_after)

    s = p_oot.argsort()
    spl = IUS(p_oot[s], poly_model_oot[s])

    res = minimize(lc_min, params, args=(time, flux, err, spl), method='leastsq') #perform minimization
    chi2 = np.sum(res.residual) / res.nfree
    rp_best, a_best, inc_best = res.params['rp'].value, res.params['a'].value, res.params['inc'].value
    t0_best = res.params['t0'].value
    per_best = res.params['per'].value
    print('Best fit parameters: rp={:.6f}; a={:.6f}; inc={:.6f}; t0={:.6f}; per={:.6f}'.format(rp_best, a_best, inc_best, t0_best, per_best))
    print('Minimization result: {}: {}; chi2={:.4f}'.format(res.success, res.message, chi2))


    phase = ((time - t0_best)/per_best)%1
    phase = np.array([p-1 if p > 0.5 else p for p in phase])

    variability_model = spl(phase)

	
    #Produce a best fit model using the minimization results
    pm_best = bm.TransitParams()
    
    pm_best.t0 = 0                 #Time of transit centre
    pm_best.per = 1.               #Orbital period = 1 as phase folded
    pm_best.rp = rp_best           #Ratio of planet to stellar radius
    pm_best.a = a_best             #Semi-major axis (units of stellar radius)
    pm_best.inc = inc_best         #Orbital Inclination [deg]
    pm_best.ecc = args.ecc                  #Orbital eccentricity (fix circular orbits)
    pm_best.w = args.w                   #Longitude of periastron [deg] (unimportant as circular orbits)
    if not args.ld == "uniform": pm_best.u = [args.u1, args.u2]            #Stellar LD coefficients
    else: pm_best.u = []
    pm_best.limb_dark=args.ld     #LD model
	
    p_best = np.linspace(-0.5, 0.5, 20000)     #Produce a model LC using 
    m_best = bm.TransitModel(pm_best, p_best)  #the output best fit parameters
    f_bm_best = m_best.light_curve(pm_best)
    variability_model = spl(p_best)        
    f_best = f_bm_best*variability_model

  #  p1 = p_best[np.where(f_bm_best < 1)[0][0]]    #Phase of first contact
  #  p4 = p_best[np.where(f_bm_best < 1)[0][-1]]   #Phase of final contact
    
  #  t_dur = (p4 - p1) * per_best * 24             #Transit duration [hours]
  #  t_depth = (1 - f_bm_best.min()) * 100         #Transit depth [percent]

    #Produce binned data set
    bw = 5 / (1440*per_best)                    #Bin width - 10 mins in units of phase
    pbin, fbin, ebin = lc_bin(phase, flux, err, bw)

    #Produce plot of data and best fit model LC
    plt.figure(figsize=(9, 7.5))
    
    plt.plot(phase, flux, 'kx', markersize=1.5)

    plt.plot(p_best, f_bm_best, 'g--', linewidth=2)
    plt.plot(p_best, variability_model, 'r--', linewidth=2)
    plt.plot(p_best, f_best, 'b-', linewidth=3)

    plt.errorbar(pbin, fbin, ebin, fmt='bo', markersize=6)

    plt.xlabel('Phase')
    plt.ylabel('Flux')
  #  plt.title('Depth: {:.4f}%;  Duration: {:4f} hours \n Period: {:.6f} days;  epoch: {:.6f} [HJD - 2450000] \n (Rp/Rs): {:.4f};  chi2: {:.4f}'.format(t_depth, t_dur, per_best, t0_best, rp_best, chi2))
#    plt.xlim((-3*p4, 3*p4))
#    plt.ylim((f_bin.min()-t_depth/200, 1+t_depth/200))
    
#    plt.savefig('/home/astro/phrvdf/NGTS_fitting/quickfit_plots/NOI_{}_lcfit.png'.format(objid))
    time1 = TIME()
    print("Time taken: {:.4f} s".format(time1-time0))

    plt.show()
    
