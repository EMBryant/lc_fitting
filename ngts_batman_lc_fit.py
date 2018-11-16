'''Script to fit the transit LC of NGTS obs'''

import numpy as np
import matplotlib.pyplot as plt
import batman as bm
from scipy.optimize import Bounds, minimize as mini
import argparse
from astropy.constants import R_sun
from scipy.stats import sem

def full_fit(X0, phase, flux, err):
	
	t0 = X0[0]
	per = X0[1]
	rp = X0[2]
	a = X0[3]
	inc = X0[4]
	ecc = X0[5]
	w = X0[6]
	u1 = X0[7]
	u2 = X0[8]
	
	pm = bm.TransitParams()
	
	pm.t0 = t0
	pm.rp = rp
	pm.a = a
	pm.inc = inc
	
	pm.per = per
	pm.ecc = ecc
	pm.w = w
	pm.u = [u1, u2]
	pm.limb_dark="quadratic"
	
	m = bm.TransitModel(pm, phase)
	
	model = m.light_curve(pm)
	
	chi_vals = np.sqrt((flux - model)**2 / err**2)
	
	fit_val = np.sum(chi_vals) / (len(chi_vals) - 1)
	
	return fit_val

def fold_fit(X0, phase, flux, err):
	
	rp = X0[0]
	a = X0[1]
	inc = X0[2]
	ecc = X0[3]
	w = X0[4]
	u1 = X0[5]
	u2 = X0[6]
	
	pm = bm.TransitParams()
	
	pm.t0 = 0
	pm.rp = rp
	pm.a = a
	pm.inc = inc
	
	pm.per = 1.
	pm.ecc = ecc
	pm.w = w
	pm.u = [u1, u2]
	pm.limb_dark="quadratic"
	
	m = bm.TransitModel(pm, phase)
	
	model = m.light_curve(pm)
	
	chi_vals = np.sqrt((flux - model)**2 / err**2)
	
	fit_val = np.sum(chi_vals) / (len(chi_vals) - 1)
	
	return fit_val

def lc_bin(time, flux, bin_width):
	
	edges = np.arange(np.min(time), np.max(time), bin_width)
	dig = np.digitize(time, edges)
	time_binned = (edges[1:] + edges[:-1]) / 2
	flux_binned = np.array([np.nan if len(flux[dig == i]) == 0 else flux[dig == i].mean() for i in range(1, len(edges))])
	err_binned = np.array([np.nan if len(flux[dig == i]) == 0 else sem(flux[dig == i]) for i in range(1, len(edges))])
	time_bin = time_binned[~np.isnan(err_binned)]
	err_bin = err_binned[~np.isnan(err_binned)]
	flux_bin = flux_binned[~np.isnan(err_binned)]	
	
	return time_bin, flux_bin, err_bin


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-fn', '--file', type=str)
	parser.add_argument('-bw', '--binwidth', type=int)
	parser.add_argument('-t0', '--epoc', type=float)
	parser.add_argument('-per', '--period', type=float)
	parser.add_argument('-rp', '--rad', type=float)
	parser.add_argument('-a', '--smaxis', type=float)

	args = parser.parse_args()
	
	data_full = np.loadtxt(args.file)
	time_full, flux_full = data_full[:, 0] - data_full[:, 0].min(), data_full[:, 3]  
	
	bw = args.binwidth / 1440  #put bin width into units of days	
	time_bin, flux_bin, err_bin = lc_bin(time_full, flux_full, bw)
	
	t0, per, rp, a = args.epoc, args.period, args.rad, args.smaxis
	
	bounds = Bounds(([t0 - 0.1, per - 0.1, rp - 0.05, a - 5, 88., 0., 0., 0., 0.]), ([t0+0.1, per+0.1, rp+0.05, a+5, 90., 1.0, 90., 1.0, 1.0]))
	
	res = mini(full_fit, [args.epoc, args.period, args.rad, args.smaxis, 89., 0., 90., 0.3, 0.2], args=(time_bin, flux_bin, err_bin), bounds=bounds)
	
	print(res.x, res.fun)

	pm = bm.TransitParams()
	
	pm.t0 = res.x[0]
	pm.per = res.x[1]
	pm.rp = res.x[2]
	pm.a = res.x[3]
	pm.inc = res.x[4]
	pm.ecc = res.x[5]
	pm.w = res.x[6]
	u1, u2 = res.x[7], res.x[8]
	pm.u = [u1, u2]
	
	pm.limb_dark="quadratic"
	
	time_model = np.arange(time_bin[0], time_bin[-1], bw/5)
	m = bm.TransitModel(pm, time_model)
	flux_model = m.light_curve(pm)

	plt.figure()

	plt.plot(time_bin, flux_bin, 'ko', markersize=3)
	plt.plot(time_model, flux_model, 'r--')
	plt.xlabel('Time [days]')
	plt.ylabel('Relative Flux')
	plt.show()

	
	
	
	
	
