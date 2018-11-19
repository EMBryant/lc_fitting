import numpy as np
import matplotlib.pyplot as plt
import batman as bm
import argparse
from scipy.optimize import Bounds, minimize as mini

def fit(X0, phase1, flux1, err1, phase2, flux2, err2):

        rp1 = X0[0]
        rp2 = X0[1]
        a = X0[2]
        inc = X0[3]
        ecc = X0[4]
        w = X0[5]
        u1 = X0[6]
        u2 = X0[7]
        u3 = X0[8]
        u4 = X0[9]
        
        pm = bm.TransitParams()
        pm2 = bm.TransitParams()
        pm.t0 = 0.
        pm2.t0 = args.epoc
        pm.rp = rp1
        pm2.rp = rp2
        pm.a = pm2.a = a
        pm.inc = pm2.inc = inc

        pm.per = pm2.per = 1.
        pm.ecc = pm2.ecc = ecc
        pm.w = pm2.w = w
        pm.u = [u1, u2]
        pm2.u = [u3, u4]
        pm.limb_dark="quadratic"
        pm2.limb_dark = "quadratic"
        m = bm.TransitModel(pm, phase1)
        m2 = bm.TransitModel(pm2, phase2)
        model1 = m.light_curve(pm)
        model2 = m2.light_curve(pm2)

        chi_vals1 = np.sqrt((flux1 - model1)**2 / err1**2)
        chi_vals2 = np.sqrt((flux2 - model2)**2 / err2**2)
        fit_val = (np.sum(chi_vals1) + np.sum(chi_vals2)) / (len(chi_vals1) + len(chi_vals2) - 1)

        return fit_val


if __name__ == "__main__":

        parser = argparse.ArgumentParser()
        parser.add_argument('fn1', type=str)
        parser.add_argument('fn2', type=str)
        parser.add_argument('lc', type=str)
        parser.add_argument('-bw', '--binwidth', type=int)
        parser.add_argument('-t0', '--epoc', type=float)
        parser.add_argument('-per', '--period', type=float)
        parser.add_argument('-rp', '--rad', type=float)
        parser.add_argument('-a', '--smaxis', type=float)

        args = parser.parse_args()


        data1 = np.loadtxt(args.fn1)
        phase_bin1, flux_bin1, err_bin1 = data1[:, 0], data1[:, 1], data1[:, 2]
        
        data2 = np.loadtxt(args.fn2)
        phase_bin2, flux_bin2, err_bin2 = data2[:, 0], data2[:, 1], data2[:, 2]


        t0, rp, a = args.epoc, args.rad, args.smaxis

        phase_fit1 = np.array(phase_bin1, dtype=float)
        phase_fit2 = np.array(phase_bin2, dtype=float)

        bounds = Bounds(([0.01, 0.01, 5., 85., 0.5, 0., 0., 0., 0., 0.]), ([1., 1., 30., 90., 0.9, 360., 1., 1., 1., 1.]))

        res = mini(fit, [rp, rp, a, 89., 0.7, 90., 0.3, 0.2, 0.3, 0.2], args=(phase_fit1, flux_bin1, err_bin1, phase_fit2, flux_bin2, err_bin2), bounds=bounds)
        print(res.fun, res.success)
        print(res.x)

        pm = bm.TransitParams()
        pm2 = bm.TransitParams()

        pm.t0 = 0.
        pm2.t0 = t0
        pm.per = pm2.per = 1.
        pm.rp = res.x[0]
        pm2.rp = res.x[1]
        pm.a = pm2.a = res.x[2]
        pm.inc = pm2.inc = res.x[3]
        pm.ecc = pm2.ecc = res.x[4]
        pm.w = pm2.w = res.x[5]
        pm.u = [res.x[6], res.x[7]]
        pm2.u = [res.x[8], res.x[9]]
        pm.limb_dark = pm2.limb_dark = "quadratic"

        phase_model1 = np.linspace(phase_fit1.min(), phase_fit1.max(), len(phase_fit1))
        phase_model2 = np.linspace(phase_fit2.min(), phase_fit2.max(), len(phase_fit2))

        m = bm.TransitModel(pm, phase_model1)
        flux_model1 = m.light_curve(pm)
        
        m2 = bm.TransitModel(pm2, phase_model2)
        flux_model2 = m2.light_curve(pm2)

        axis_font = {'fontname':'DejaVu Sans', 'size':'20'}

        fig = plt.figure()
        
        ax1 = fig.add_subplot(211)

        ax1.errorbar(phase_fit1, flux_bin1, yerr=err_bin1, marker='o', color='black', linestyle='none', markersize=3)
        ax1.errorbar(phase_fit2, flux_bin2, yerr=err_bin2, marker='o', color='blue', linestyle='none', markersize=3)
        ax1.plot(phase_model1, flux_model1, 'r--')
        ax1.plot(phase_model2, flux_model2, 'g--')
        ax1.set_xlabel('Phase', **axis_font)
        ax1.set_ylabel('Relative Flux', **axis_font)
        ax1.set_title('NOI 104155; Period = 12.1804 days \n a={:.2f}; inc={:.2f}$^o$; ecc={:.2f}; w={:.2f}$^o$; $\chi^2$={:.2f}'.format(res.x[2], res.x[3], res.x[4], res.x[5], res.fun), **axis_font)

        ax2 = fig.add_subplot(224)

        ax2.errorbar(phase_fit1, flux_bin1, yerr=err_bin1, marker='o', color='black', linestyle='none', markersize=3)
        ax2.plot(phase_model1, flux_model1, 'r--')
        ax2.set_xlabel('Phase', **axis_font)
        ax2.set_title('Rp = {:.2f}'.format(res.x[0]), **axis_font)
        
        ax3 = fig.add_subplot(223)
        
        ax3.errorbar(phase_fit2, flux_bin2, yerr=err_bin2, marker='o', color='blue', linestyle='none', markersize=3)
        ax3.plot(phase_model2, flux_model2, 'g--')
        ax3.set_xlabel('Phase', **axis_font)
        ax3.set_ylabel('Relative Flux', **axis_font)
        ax3.set_title('Rp = {:.2f}'.format(res.x[1]), **axis_font)
        
        plt.show()
