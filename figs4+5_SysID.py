""" Script to produce Figs. 4, 5 and S6-9 """

from ofc import System, parmap
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

# Okabe & Ito's colorblind friendly palette
colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
plt.rc('axes', prop_cycle=plt.cycler('color', colors))
plt.rc('font', size=18)
plt.rc('legend', **{'fontsize': 12})


def run(system, eta, etaK, momentum, sigma, delay, episodes=10000, runs=20, loop='closed'):
    x0 = [0] * system.m
    x0[0] = -1

    def foo(seed):
        np.random.seed(seed)
        if system.m == system.n:
            Ahat, Bhat = [.1*np.random.randn(*a_.shape) for a_ in (system.A, system.B)]
            Chat, Lhat = [.5*np.eye(system.m) + .5*np.random.rand(*a.shape)
                          for a in (system.C, system.C.T)]
        else:
            Ahat, Bhat, Chat, Lhat = [.1*np.random.randn(*a_.shape) for a_ in
                                      (system.A, system.B, system.C, system.C.T)]
            while np.any(np.linalg.eigvals(Lhat.dot(Chat)) <= .01):
                Lhat = .1*np.random.randn(*Chat.T.shape)
        if loop == 'closed':
            return system.PGwithSysID(seed, eta, etaK, momentum, sigma, delay,
                                      episodes, ABCLhat=(Ahat, Bhat, Chat, Lhat), x0=x0)
        else:
            sysIdresult = system.SysID(Ahat, Bhat, Chat, Lhat, eta, delay, episodes//2,
                                       x0=x0, init_seed=seed)
            return system.PGafterSysID(seed, etaK, momentum, sigma, delay, episodes//2, x0=x0,
                                       ABCLhat=sysIdresult[:4]) + (sysIdresult[4],)
    Js = parmap(foo, range(runs))
    J, J2, mse = [np.array([j[i] for j in Js]) for i in range(3)]
    return J, J2, mse


s0 = System(A=np.array([[1, 1], [0, 1]]),
            B=np.array([[0], [1]]),
            C=np.eye(2),
            V=.01 * np.eye(2),
            W=np.diag([.04, .25]),
            Q=np.array([[1, 0], [0, 0]]),
            R=np.ones((1, 1)),
            T=11)

s1 = System(A=np.array([[1, 1], [0, 1]]),
            B=np.array([[0], [1]]),
            C=np.array([[1, 0], [0, -1], [.5, .5]]),
            V=.01 * np.eye(2),
            W=np.array([[.04, .09, 0], [.09, .25, 0], [0, 0, .04]]),
            Q=np.array([[1, 0], [0, 0]]),
            R=np.ones((1, 1)),
            T=11)

s2 = System(A=np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]]),
            B=np.array([[0], [1], [0]]),
            C=np.array([[1, 0, 0], [0, -1, 0], [.5, .5, 0]]),
            V=.01 * np.eye(3),
            W=np.array([[.04, .09, 0], [.09, .25, 0], [0, 0, .04]]),
            Q=np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
            R=np.ones((1, 1)),
            T=11)


for (momentum, sigma) in ((.99, .2), (.99, .05), (.99, .1), (.99, .5),
                          (0, .2), (.9, .2), (.9995, .2)):
    for loop in ('closed', 'open'):
        results = {}
        for ss in ((0, 1, 2) if (momentum == .99 and sigma == .2) else (0,)):
            s = (s0, s1, s2)[ss]
            for delay in (1, 2, 3):
                try:
                    params = np.load('results/LQG_%s_delay%g_sigma%g_momentum%g%s' %
                                    (loop, delay, sigma, momentum,
                                    ('_C=I.npy', '.npy', '_m=3.npy')[ss]),
                                    allow_pickle=True).item().best_params
                except FileNotFoundError:
                    print("Optimal hyperparameters not found. Run hyperopt.sh first.")                    
                eta, etaK = params['eta'], params['etaK']
                J, J2, mse = run(s, eta, etaK, momentum, sigma, delay=delay, loop=loop)
                results['delay%g' % delay] = {'J': J, 'J2': J2, 'mse': mse}

            try:
                asymp = np.load('results/LQG_delay' + ('_C=I.npy', '.npy', '.npy')[ss],
                                allow_pickle=True).item()['current2']['performance']
            except FileNotFoundError:
                print("Output file of fig3_delay.py not found. Run fig3_delay.py first.")
            for j, perf in enumerate(('cost', 'mse')[:(2 if loop == 'open' else 1)]):
                plt.figure(figsize=(6, 4))
                for delay in (1, 2, 3):
                    if j:
                        d = results['delay%g' % delay]['mse']
                    else:
                        d = np.hstack([results['delay%g' % delay]['J'],
                                       results['delay%g' % delay]['J2']])
                    plt.fill_between(range(d.shape[1]),
                                     median_filter(d.mean(0)+d.std(0)/np.sqrt(len(d)-1), 51),
                                     median_filter(d.mean(0)-d.std(0)/np.sqrt(len(d)-1), 51),
                                     color='C%g' % (delay-1), alpha=.3)
                    plt.plot(median_filter(d.mean(0), 51),
                             label=str(delay), c='C%g' % (delay-1))
                plt.legend(title='Delay')
                plt.xlabel('Episodes')
                plt.ylabel(('Cost', 'MSE')[j])
                for delay in (1, 2, 3):
                    plt.axhline((asymp[1], asymp[0])[j][delay].mean(),
                                ls='--', c='C%g' % (delay-1))
                if j == 0:
                    plt.axvline(10000 if loop == 'closed' else 5000,
                                ls='--', c='k', zorder=-11)
                    plt.ylim(3, 43)
                else:
                    plt.ylim(0, 1.8)
                plt.tight_layout(pad=.05)
                plt.savefig('fig/LQG_%s_%s_sigma%g_momentum%g%s' %
                            (perf, loop, sigma, momentum, ('_C=I.pdf', '.pdf', '_m=3.pdf')[ss]))
