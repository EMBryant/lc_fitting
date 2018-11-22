'''Script to fit the transit LC of NGTS obs'''

import numpy as np
import matplotlib.pyplot as plt
import batman as bm
from scipy.optimize import Bounds, minimize as mini
import argparse
from astropy.constants import R_sun
from scipy.stats import sem

def bm_lc_model(t, t0, per, rp, a, inc=89., ecc=0., w=90., u1=0.3, u2=0.2, ld="quadratic"):
	
	pm = bm.TransitParams()

	pm.t0 = t0
	pm.per = per
	pm.rp = rp
	pm.a = a
	pm.inc = inc
	pm.ecc = ecc
	pm.w = w
	pm.u = [u1, u2]
	pm.limb_dark = ld

	m = bm.TransitModel(pm, t)
	flux = m.light_curve(pm)
	return flux

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
	
	model = bm_lc_model(phase, 0.25, 1., rp, a, inc, ecc, w, u1, u2)
	
	chi_vals = np.sqrt((flux - model)**2 / err**2)
	
	fit_val = np.sum(chi_vals) / (len(chi_vals) - 1)
	
	return fit_val

def fold_fit(X0, phase, flux, err):
	
        t0 = X0[0]
        rp = X0[1]
        a = X0[2]
        inc = X0[3]
        ecc = X0[4]
        w = X0[5]
        u1 = X0[6]
        u2 = X0[7]

        model = bm_lc_model(phase, t0, 1., rp, a, inc, ecc, w, u1, u2)

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
	parser.add_argument('fn', type=str, help="Name of data file")
	parser.add_argument('fit', type=str, help="Type of fit to perform")
	parser.add_argument('-bw', '--binwidth', type=int, default=5, help="Bin width in units of minutes")
	parser.add_argument('-t0', '--epoc', type=float, default=0., help="First guess time of first transit")
	parser.add_argument('-per', '--period', type=float, default=1., help="First guess of orbital period")
	parser.add_argument('-rp', '--rad', type=float, default=0.1, help="First guess of planet:star radius ratio")
	parser.add_argument('-a', '--smaxis', type=float, default=10., help="First guess of sm axis (units of stellar radius")
	parser.add_argument('-inc', '--inclination', type=float, default=89.)
	parser.add_argument('-e', '--ecc', type=float, default=0.)
	parser.add_argument('-w', '--omega', type=float, default=90.)
	parser.add_argument('-fp', '--flux', type=float, default=1.)

	args = parser.parse_args()
	
	t0, per, rp, a, inc, ecc, w, fp = args.epoc, args.period, args.rad, args.smaxis, args.inclination, args.ecc, args.omega, args.flux
	
	if args.fit == 'full':
	
		data_full = np.loadtxt(args.fn)
		time_full, flux_full = data_full[:, 0] - data_full[:, 0].min(), data_full[:, 3]  
	
		bw = args.binwidth / 1440  #put bin width into units of days	
		time_bin, flux_bin, err_bin = lc_bin(time_full, flux_full, bw)
	
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
	
	if args.fit == 'pf':

		data = np.loadtxt(args.fn)
		PHASE, flux, err = data[:, 0], data[:, 1], data[:, 2]
		phase = np.array(PHASE, dtype=float)-0.176
		
		bounds = Bounds(([t0-0.005, 0.1, 5., 85., 0.1, 0., 0., 0.]), ([t0+0.005, 1., 35., 90., 0.9, 45., 1., 1.]))

		res = mini(fold_fit, [t0, rp, a, inc, ecc, w, 0.3, 0.2], args=(phase, flux, err), bounds=bounds)

		print(r'Minimization success is {} with $\chi^2$={:.2f}'.format(res.success, res.fun))
		print("t0 = {:.4f}; rp = {:.4f}; a = {:.4f}".format(res.x[0], res.x[1], res.x[2]))
		print("inc = {:.4f}; ecc = {:.4f}; w = {:.4f}; u = [{:.4f}, {:.4f}]".format(res.x[3], res.x[4], res.x[5], res.x[6], res.x[7]))

		phase_plot = np.array([p - 1. if p > 0.75 else p for p in phase])
		phase_model = phase_plot[np.argsort(phase_plot)]
		
		flux_model = bm_lc_model(phase_model, res.x[0], 1., res.x[1], res.x[2], res.x[3], res.x[4], res.x[5], res.x[6], res.x[7])
		t1, t4 = phase_model[np.where(flux_model < 1)[0][0]], phase_model[np.where(flux_model < 1)[0][-1]]
		T_dur = (t4 - t1) * per * 24
		depth = 1 - flux_model.min()
		
		p_bin, f_bin, e_bin = lc_bin(phase_plot, flux, args.binwidth/(per * 1440))

		axis_font = {'fontname':'DejaVu Sans', 'size':'20'}
		
		plt.figure()

		plt.plot(phase_plot, flux, marker='o', color='grey', linestyle='none', markersize=0.5)
		plt.plot(p_bin, f_bin, 'ko', markersize=5)
		plt.plot(phase_model, flux_model, 'r--', linewidth=2)
		plt.xlabel('Phase [days]', **axis_font)
		plt.ylabel('Relative Flux', **axis_font)
		plt.title('NOI 104155; Period = {:.4f} days; Transit 2; Tdur={:.4f} hours; depth={:.1f}% \n t0={:.4f}; Rp={:.4f}; a={:.4f}; inc={:.4f}$^o$; ecc={:.4f}; w={:.4f}$^o$; $\chi^2$={:.4f}'.format(per, T_dur, depth*100, res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.x[5], res.fun), **axis_font)

		plt.show()
	
	
