import numpy as np
import matplotlib.pyplot as plt
import batman as bm
import argparse
from scipy.optimize import Bounds, minimize as mini
from scipy.stats import sem

def fit_secondary(X0, p, f, e):
		
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

def fit_db(X0, p1, f1, e1, p2, f2, e2):
	
		pm1 = bm.TransitParams()
		pm2 = bm.TransitParams()
		
		t01, t02 = X0[0], X0[1]
		rp1 = X0[2]
		rp2 = 1/rp1
		fp1 = X0[3]
		fp2 = 1/fp1
		a1, a2 = X0[4], X0[5]
		inc = X0[6]
		ecc = X0[7]
		w1 = X0[8]
		w2 = w1+180
		u1, u2 = X0[9], X0[10]
		u3, u4 = X0[11], X0[12]
		
		pm1.t0 = t01
		pm1.per = 1.
		pm1.rp = rp1
		pm1.a = a1
		pm1.inc = inc
		pm1.ecc = ecc
		pm1.w = w1
		pm1.u = [u1, u2]
		pm1.limb_dark="quadratic"
		
		pm2.t0 = t02
		pm2.per = 1.
		pm2.rp = rp2
		pm2.a = a2
		pm2.inc = inc
		pm2.ecc = ecc
		pm2.w = w2
		pm2.u = [u3, u4]
		pm2.limb_dark="quadratic"
		
		m1 = bm.TransitModel(pm1, p1)
		m2 = bm.TransitModel(pm2, p2)
		
		F1 = (m1.light_curve(pm1) + fp1) / (1 + fp1)
		F2 = (m2.light_curve(pm2) + fp2) / (1 + fp2)
		
		fit_vals1 = np.sqrt((f1 - F1)**2 / e1**2)
		fit_vals2 = np.sqrt((f2 - F2)**2 / e2**2)
		
		fit_val = (np.sum(fit_vals1) + np.sum(fit_vals2)) / (len(fit_vals1) + len(fit_vals2) - 1)
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
    parser.add_argument('fit', type=str)
    parser.add_argument('-f2', type=str)
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
    
    if args.fit == "sec":
		
        data = np.loadtxt(args.fn)
        phase, flux, err = data[:, 0], data[:, 1], data[:, 2]
        
        t0, per, rp, a, inc, e, w, fp = args.epoc, args.period, args.rad, args.smaxis, args.inclination, args.ecc, args.omega, args.flux

        phase_fit = np.array(phase, dtype=float)

        bounds = Bounds(([28., 85., 0.53]), ([33., 90., 0.54]))

        res = mini(fit_secondary, [a, inc, e], args=(phase_fit, flux, err), bounds=bounds)
        
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

    if args.fit == "db":
		
        data1 = np.loadtxt(args.fn)
        data2 = np.loadtxt(args.f2)
        
        PHASE1, flux1, err1 = data1[:, 0], data1[:, 1], data1[:, 2]
        phase1 = np.array(PHASE1, dtype=float)
        
        PHASE2, flux2, err2 = data2[:, 0], data2[:, 1], data2[:, 2]
        phase2 = np.array(PHASE2, dtype=float)
        
        t01, per, rp, a, inc, e, w, fp = args.epoc, args.period, args.rad, args.smaxis, args.inclination, args.ecc, args.omega, args.flux
        t02 = t01
        bounds = Bounds(([t01-0.005, t02-0.005, 0.1, 0.1, 2, 2, 80., 0.4, 0., 0., 0., 0., 0.]), ([t01+0.005, t02+0.005, 10, 10, 50, 35, 90., 0.95, 50., 1., 1., 1., 1.]))		
        res = mini(fit_db, [t01, t02, rp, fp, a, a, inc, e, w, 0.3, 0.2, 0.3, 0.2], args=(phase1, flux1, err1, phase2, flux2, err2), bounds=bounds)
        
        print("Minimization success is {} with chi2={:.3f}".format(res.success, res.fun))
        print("t01={:.4f}; t02={:.4f}; rp={:.4f}; fp={:.4f}".format(res.x[0], res.x[1], res.x[2], res.x[3]))
        print("a1={:.4f}; a2={:.4f}; inc={:.4f}; ecc={:.4f}; w={:.4f}".format(res.x[4], res.x[5], res.x[6], res.x[7], res.x[8]))
        print("u1=[{:.3f}, {:.3f}]; u2=[{:.3f}, {:.3f}]".format(res.x[9], res.x[10], res.x[11], res.x[12]))
        
        pm1 = bm.TransitParams()
        pm2 = bm.TransitParams()
        
        t01, t02 = res.x[0], res.x[1]
        rp1 = res.x[2]
        rp2 = 1/rp1
        fp1 = res.x[3]
        fp2 = 1/fp1
        a1, a2 = res.x[4], res.x[5]
        inc = res.x[6]
        ecc = res.x[7]
        w1 = res.x[8]
        w2 = w1+180
        u1, u2 = res.x[9], res.x[10]
        u3, u4 = res.x[11], res.x[12]
        
        pm1.t0 = t01
        pm1.per = 1.
        pm1.rp = rp1
        pm1.a = a1
        pm1.inc = inc
        pm1.ecc = ecc
        pm1.w = w1
        pm1.u = [u1, u2]
        pm1.limb_dark="quadratic"
        
        pm2.t0 = t02
        pm2.per = 1.
        pm2.rp = rp2
        pm2.a = a2
        pm2.inc = inc
        pm2.ecc = ecc
        pm2.w = w2
        pm2.u = [u3, u4]
        pm2.limb_dark="quadratic"
        
        phase_model1 = np.linspace(phase1.min(), phase1.max(), len(phase1))
        phase_model2 = np.linspace(phase2.min(), phase2.max(), len(phase2))
        
        m1 = bm.TransitModel(pm1, phase_model1)
        m2 = bm.TransitModel(pm2, phase_model2)
        
        F1 = (m1.light_curve(pm1) + fp1) / (1 + fp1)
        F2 = (m2.light_curve(pm2) + fp2) / (1 + fp2)
 #       flux_model = F1*F2
        
        bw = args.binwidth / (1440 * per)
        pb1, fb1, eb1 = lc_bin(phase1, flux1, bw)
        pb2, fb2, eb2 = lc_bin(phase2, flux2, bw)
        
        fig = plt.figure()
        
        ax1 = fig.add_subplot(121)
        
        ax1.plot(phase1, flux1, 'ko', color='gray', markersize=1.5)
        ax1.errorbar(pb1, fb1, yerr=eb1, marker='o', color='black', linestyle='none', markersize=3)
        ax1.plot(phase_model1, F1, 'r--')
        
        ax1.set_xlabel('Phase')
        ax1.set_ylabel('Relative Flux')
        ax1.set_title('Primary Transit: t0={:.4f}; rp={:.4f}; fp={:.4f} \n a={:.4f}; inc={:.4f}; ecc={:.4f}; w={:.4f} '.format(t01, rp1, fp1, a1, inc, ecc, w1))
        
        ax2 = fig.add_subplot(122)
        
        ax2.plot(phase2, flux2, 'ko', color='gray', markersize=1.5)
        ax2.errorbar(pb2, fb2, yerr=eb2, marker='o', color='black', linestyle='none', markersize=3)
        ax2.plot(phase_model2, F2, 'r--')
        
        ax2.set_xlabel('Phase')
        ax2.set_title('Secondary Transit: t0={:.4f}; rp={:.4f}; fp={:.4f} \n a={:.4f}; inc={:.4f}; ecc={:.4f}; w={:.4f} '.format(t02, rp2, fp2, a2, inc, ecc, w2))
        
        plt.show()
       
        
        
        
        
