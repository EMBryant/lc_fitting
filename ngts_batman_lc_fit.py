'''Script to fit the transit LC of NGTS obs'''

import numpy as np
import matplotlib.pyplot as plt
import batman as bm
from scipy.optimize import Bounds, minimize as mini
import argparse
from astropy.constants import R_sun
from scipy.stats import sem

def period_fit_old(X0, rp, a, inc, ecc, w, u1, u2, phase, flux, err):
	
	t0 = X0[0]
	per = X0[1]

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

def period_fit(X0, time, flux, err):
	
	t0 = X0[0]
	per = X0[1]	
	rp = X0[2]
	a = X0[3]
	inc = X0[4]
	ecc = X0[5]
	w = X0[6]
	u1 = X0[7]
	u2 = X0[8]
	
	phase = (((time - t0)/per)%1)+0.25

	pm = bm.TransitParams()
	
	pm.t0 = 0.25
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

def fold_fit(X0, phase, flux, err):
	
        rp = X0[0]
        a = X0[1]
        inc = X0[2]
        ecc = X0[3]
        w = X0[4]
        u1 = X0[5]
        u2 = X0[6]

        pm = bm.TransitParams()

        pm.t0 = args.epoc
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
	parser.add_argument('fn', type=str)
	parser.add_argument('lc', type=str)
	parser.add_argument('-bw', '--binwidth', type=int)
	parser.add_argument('-t0', '--epoc', type=float)
	parser.add_argument('-per', '--period', type=float)
	parser.add_argument('-rp', '--rad', type=float)
	parser.add_argument('-a', '--smaxis', type=float)

	args = parser.parse_args()

	if args.lc == 'full':
	
		data_full = np.loadtxt(args.fn)
		time_full, flux_full = data_full[:, 0] - data_full[:, 0].min(), data_full[:, 3]  
	
		bw = args.binwidth / 1440  #put bin width into units of days	
		time_bin, flux_bin, err_bin = lc_bin(time_full, flux_full, bw)
	
		t0, per, rp, a = args.epoc, args.period, args.rad, args.smaxis
	
		bounds_full = Bounds(([t0-0.1, per-0.05, rp-0.3, a-5, 85., 0., 0., 0., 0.]), ([t0+0.1, per+0.05, rp+0.3, a+5, 90., 1.0, 90., 1.0, 1.0]))
		res = mini(fold_fit, [t0, per, rp, a, 89., 0., 90., 0.3, 0.2], args=(time_bin, flux_bin, err_bin), bounds=bounds_full)
	
		print(res.x, res.fun)	
		print(res.success)
		pm = bm.TransitParams()
	
		pm.t0 = 0.25
		pm.per = 1.
		pm.rp = res.x[2]
		pm.a = res.x[3]
		pm.inc = res.x[4]
		pm.ecc = res.x[5]
		pm.w = res.x[6]
		u1, u2 = res.x[7], res.x[8]
		pm.u = [u1, u2]
	
		phase = (((time_bin - res.x[0])/res.x[1])%1) + 0.25
	
		pm.limb_dark="quadratic"
	
		m = bm.TransitModel(pm, phase)
		flux_model = m.light_curve(pm)
	
		plt.figure()

		plt.plot(phase, flux_bin, 'ko', markersize=3)
		plt.plot(phase, flux_model, 'rx')
		plt.xlabel('Time [days]')
		plt.ylabel('Relative Flux')
		plt.show()

	
	if args.lc == 'pf':

		data = np.loadtxt(args.fn)
		phase_bin, flux_bin, err_bin = data[:, 0], data[:, 1], data[:, 2]
		
		t0, rp, a = args.epoc, args.rad, args.smaxis

		phase_fit = np.array(phase_bin, dtype=float)

		bounds = Bounds(([0.01, 5., 85., 0.5, 0., 0., 0.]), ([1., 30., 90., 0.9, 360., 1., 1.]))

		res = mini(fold_fit, [rp, a, 89., 0.7, 90., 0.3, 0.2], args=(phase_fit, flux_bin, err_bin), bounds=bounds)	
		print(res.fun, res.success)
		print(res.x)

		pm = bm.TransitParams()
	
		pm.t0 = t0
		pm.per = 1.
		pm.rp = res.x[0]
		pm.a = res.x[1]
		pm.inc = res.x[2]
		pm.ecc = res.x[3]
		pm.w = res.x[4]
		pm.u = [res.x[5], res.x[6]]
		pm.limb_dark = "quadratic"
		
		phase_model = np.linspace(phase_fit.min(), phase_fit.max(), len(phase_fit))
		
		m = bm.TransitModel(pm, phase_model)
		flux_model = m.light_curve(pm)
		
		axis_font = {'fontname':'DejaVu Sans', 'size':'20'}

		plt.figure()

		plt.errorbar(phase_fit, flux_bin, yerr=err_bin, marker='o', color='black', linestyle='none', markersize=3)
		plt.plot(phase_model, flux_model, 'r--')
		plt.xlabel('Phase [days]', **axis_font)
		plt.ylabel('Relative Flux', **axis_font)
		plt.title('NOI 104155; Period = 12.1804 days; Transit 2 \n Rp={:.2f}; a={:.2f}; inc={:.2f}$^o$; ecc={:.2f}; w={:.2f}$^o$; $\chi^2$={:.2f}'.format(res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.fun), **axis_font)

		plt.show()
	
	
