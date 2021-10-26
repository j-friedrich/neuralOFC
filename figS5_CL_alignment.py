""" Script to produce Fig. S5 """

from ofc import System, parmap
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import median_filter
from scipy.stats import ttest_ind

# Okabe & Ito's colorblind friendly palette
colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
          "#0072B2", "#D55E00", "#CC79A7", "#000000"]
plt.rc('axes', prop_cycle=plt.cycler('color', colors))
plt.rc('font', size=18)
plt.rc('legend', **{'fontsize': 14})

s0 = System(A=np.array([[1, 1], [0, 1]]),
            B=np.array([[0], [1]]),
            C=np.eye(2),
            V=.01 * np.eye(2),
            W=np.diag([.04, .25]),
            Q=np.array([[1, 0], [0, 0]]),
            R=np.ones((1, 1)),
            T=11)


# EVs of initial L*pseudoinv(C') are 1-2*lam and 1
def foo(lam):
    L0 = np.array([[1-lam, lam], [lam, 1-lam]], dtype=float)

    def objective(lr, seed):
        try:
            return s0.SysID(s0.A.astype(float), s0.B.astype(float), s0.C.astype(float), L0,
                            (0, 0, 0, lr), init_seed=seed, episodes=10000)[-1].mean()
        except:
            return 1000
    opt = minimize(lambda lr: np.mean(parmap(lambda seed: objective(lr, seed*10000),
                                             range(20))), .05 if lam < .5 else .0001)
    return opt, np.array(parmap(lambda seed: s0.SysID(
        s0.A.astype(float), s0.B.astype(float), s0.C.astype(float), L0,
        (0, 0, 0, opt.x), init_seed=seed*10000, episodes=10000)[-1], range(20)))


lams = np.hstack([np.arange(0, .49, .05), np.arange(.46, .491, .01)])
try:
    res, res2 = list((np.load('results/CL_alignment.npz', allow_pickle=True)[r]
                      for r in ('res', 'res2')))
except:
    res = list(map(foo, lams))
    res2 = list(map(foo, (.5, .51)))
    np.savez('results/CL_alignment.npz', res=res, res2=res2)

plt.figure(figsize=(6, 4))
plt.errorbar(1-2*lams, [r[1].mean() for r in res],
             [r[1].mean(1).std()/np.sqrt(19) for r in res], label='all 10000 epochs')
plt.errorbar(1-2*lams, [r[1][:, -100:].mean() for r in res],
             [r[1][:, -100:].mean(1).std()/np.sqrt(19) for r in res], label='last 100 epochs')
plt.legend(title='MSE averaged over')
plt.xlabel(r'Minimal eigenvalue $\lambda_1$ of initial $LC^{\top+}$')
plt.ylabel('MSE')
plt.tight_layout(pad=.05)
plt.savefig('fig/CL_alignment_avg.pdf')

pvalues = np.array([[ttest_ind(r[1][:, -100:].mean(1), rr[1]
                               [:, -100:].mean(1)).pvalue for r in res] for rr in res])
print(pvalues.min())
print(pvalues[:-1,:-1].min())


plt.figure(figsize=(6, 4))
for i, (ev, d) in enumerate(((1, res[0][1]), (.04, res[-2][1]), (.02, res[-1][1]),
                             (0, res2[0][1]), (-.02, res2[1][1]))[::-1]):
    plt.fill_between(range(d.shape[1]),
                     median_filter(d.mean(0)+d.std(0)/np.sqrt(len(d)-1), 101),
                     median_filter(d.mean(0)-d.std(0)/np.sqrt(len(d)-1), 101),
                     alpha=.3, color='C%g' % i)
    plt.plot(median_filter(d.mean(0), 101), label=str(ev), c='C%g' % i)
plt.legend(title='Minimal eigenvalue $\lambda_1$', ncol=3)
plt.xlabel('Episodes')
plt.ylabel('MSE')
plt.tight_layout(pad=.05)
plt.savefig('fig/CL_alignment.pdf')
