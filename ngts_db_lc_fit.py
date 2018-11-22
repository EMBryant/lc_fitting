import numpy as np
import matplotlib.pyplot as plt
import batman as bm
import argparse
from scipy.optimize import Bounds, minimize as mini
from scipy.stats import sem

def db_fit(X0, p, f, e):
		
		pm = bm.TransitParams()
		pm.t0 = t0
		pm.per = 1.
		pm.rp = rp
		pm.a = X0[0]
		pm.inc = X0[1]
		pm.ecc = X0[2]
		pm.w = w
		pm.u=[0.3, 0.2]
		pm.limb_dark="quadratic"
		pm.fp = fp
		
		m = bm.TransitModel(pm, p)
		pm.t_secondary = m.get_t_secondary(pm)
		m2 = bm.TransitModel(pm, p, transittype="secondary")
		f1 = (m.light_curve(pm) + 1) / 2
		f2 = m2.light_curve(pm) / 2
		
		F = f1 * f2
		
		vals = np.sqrt((f-F)**2/e**2)
		fit_val = np.sum(vals)/(len(vals)-1)
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
        parser.add_argument('-bw', '--binwidth', type=int, default=5)
        parser.add_argument('-t0', '--epoc', type=float, default=0.)
        parser.add_argument('-per', '--period', type=float, default=1.)
        parser.add_argument('-rp', '--rad', type=float, default=0.1)
        parser.add_argument('-a', '--smaxis', type=float, default=10.)
        parser.add_argument('-inc', '--inclination', type=float, default=89.)
        parser.add_argument('-e', '--ecc', type=float, default=0.)
        parser.add_argument('-w', '--omega', type=float, default=90.)
        parser.add_argument('-fp', '--flux', type=float, default=1.)

        args = parser.parse_args()

        data = np.loadtxt(args.fn)
        phase, flux, err = data[:, 0], data[:, 1], data[:, 2]
        
        t0, per, rp, a, inc, e, w, fp = args.epoc, args.period, args.rad, args.smaxis, args.inclination, args.ecc, args.omega, args.flux

        phase_fit = np.array(phase, dtype=float)

        bounds = Bounds(([28., 85., 0.53]), ([33., 90., 0.54]))

        res = mini(db_fit, [a, inc, e], args=(phase_fit, flux, err), bounds=bounds)
        
        print("Minimization success is: {} with $\chi^2$={:.2f}".format(res.success, res.fun))
        print("a = {:.4f}; inc = {:.4f}; ecc = {:.4f}".format(res.x[0], res.x[1], res.x[2]))
#        print("inc = {:.4f}; ecc = {:.4f}; w = {:.4f}; u = [{:.4f}, {:.4f}]".format(res.x[3], res.x[4], res.x[5], res.x[6], res.x[7]))

        pm = bm.TransitParams()

        pm.t0 = t0
        pm.per = 1.
        pm.rp = rp
        pm.a = res.x[0]
        pm.inc = res.x[1]
        pm.ecc = res.x[2]
        pm.w = w
        pm.u = [0.3, 0.2]
        pm.limb_dark = "quadratic"

        phase_model = np.linspace(phase_fit.min(), phase_fit.max(), len(phase_fit))

        m = bm.TransitModel(pm, phase_model)
        flux_model1 = (m.light_curve(pm) + fp) / (1+fp)
        
        pm.fp = fp
        pm.t_secondary = m.get_t_secondary(pm)
        
        m2 = bm.TransitModel(pm, phase_model, transittype="secondary")
        flux_model2 = m2.light_curve(pm) / (1+fp)
        
        FLUX = flux_model1 * flux_model2
        
        bw = args.binwidth / (1440 * per)
        pb, fb, eb = lc_bin(phase, flux, bw)
        
        axis_font = {'fontname':'DejaVu Sans', 'size':'20'}

        fig = plt.figure()
        
        ax1 = fig.add_subplot(211)

        ax1.plot(phase, flux, marker='o', color='grey', linestyle='none', markersize=0.5)
        ax1.errorbar(pb, fb, yerr=eb, marker='o', color='black', linestyle='none', markersize=3)
        ax1.plot(phase_model, FLUX, 'r--')
        ax1.set_xlabel('Phase', **axis_font)
        ax1.set_ylabel('Relative Flux', **axis_font)
        ax1.set_title('NOI 104155; Period = 12.1804 days \n a={:.4f}; inc={:.4f}$^o$; ecc={:.4f}; w={:.2f}$^o$; $\chi^2$={:.4f}'.format(res.x[0], res.x[1], res.x[2], w, res.fun), **axis_font)

        ax2 = fig.add_subplot(223)

        ax2.plot(phase, flux, marker='o', color='gray', linestyle='none', markersize=0.5)
        ax2.errorbar(pb, fb, yerr=eb, marker='o', color='black', linestyle='none', markersize=3)
        ax2.plot(phase_model, FLUX, 'r--')
        ax2.set_xlabel('Phase', **axis_font)
        ax2.set_ylabel('Relative Flux', **axis_font)

        ax3 = fig.add_subplot(224)
        
        ax3.plot(phase, flux, marker='o', color='gray', linestyle='none', markersize=0.5)
        ax3.errorbar(pb, fb, yerr=eb, marker='o', color='black', linestyle='none', markersize=3)
        ax3.plot(phase_model, FLUX, 'r--')
        ax3.set_xlabel('Phase', **axis_font)
        
        plt.show()
