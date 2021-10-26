""" Script to produce Fig. 3 """

from ofc import System, parmap
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from scipy.optimize import minimize

# Okabe & Ito's colorblind friendly palette
colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
plt.rc('axes', prop_cycle=plt.cycler('color', colors))
plt.rc('font', size=18)
plt.rc('legend', **{'fontsize': 12})


def objective_kalman(system):
    """Run 10000 episodes of LQG w/o delay.

    Parameters
    ----------
    system : System
        instance of class System

    Returns
    -------
    MSEs, costs
    """
    samples = 10000
    splits = 10

    def foo(seed, samples=samples//splits):
        cost = np.zeros(samples)
        mse = np.zeros(samples)
        for i, seed in enumerate(range(seed, seed+samples)):
            U, X, Y, Xpre, Xpost = system.LQG(seed, delay=0)
            mse[i] = np.mean((Y-Xpre.dot(system.C.T))**2)
            cost[i] = np.trace(X.T.dot(X).dot(system.Q)) + np.trace(U.T.dot(U).dot(system.R))
        return mse, cost
    tmp = parmap(foo, range(0, samples, samples//splits))
    mse, cost = [np.concatenate([q[i] for q in tmp]) for i in (0, 1)]
    return mse, cost


def objective(system, ABCLhat, delay=1, full=False, update_current=False,
              actor='model', train=False):
    """Run 10000 episodes using estimates for matrices A, B, C, L.

    Parameters
    ----------
    system : System
        instance of class System
    ABCLhat : tuple of ndarrays
        The estimates for matrices A, B, C, and L.
    delay : int
        delay >= 1
    full : bool
        True:  full output: MSEs, costs
        False: output mean of the MSEs
    update_current : bool
        True:  Action U[t] is based on Xhat[t], which is updated using past
               prediction error (Y[t-delay]-C Xhat[t-delay])
        False: Action U[t] is based on Xpred[t], which is based on Xhat[t+1-delay]
    actor : string, optional
        equation for U[t], default is the optimal LQG controller "-K[t].dot(Xpost[t])"
    train : bool
        training (True) or validation (False)

    Returns
    -------
    see argument 'full'
    """
    Chat = ABCLhat[2]

    samples = 10000
    splits = 10
    init_seed = 10000 if train else 0

    def foo(seed, samples=samples//splits):
        cost = np.zeros(samples)
        mse = np.zeros(samples)
        for i, seed in enumerate(range(seed, seed+samples)):
            if update_current:
                act = "-K[t].dot(Xhat[t])" if actor == 'model' else actor
                U, X, Y, _, _, Xhat, _ = system.LQG(
                    seed, ABCLhat=ABCLhat, delay=delay, actor=act)
                mse[i] = np.mean((Y-Xhat.dot(Chat.T))**2)
            else:
                act = "-K[t].dot(Xpred[t])" if actor == 'model' else actor
                U, X, Y, _, _, _, Xpred = system.LQG(
                    seed, ABCLhat=ABCLhat, delay=delay, update_current=False, actor=act)
                mse[i] = np.mean((Y-Xpred.dot(Chat.T))**2)
            cost[i] = np.trace(X.T.dot(X).dot(system.Q)) + np.trace(U.T.dot(U).dot(system.R))
        return mse, cost
    tmp = parmap(foo, range(init_seed, init_seed+samples, samples//splits))
    mse, cost = [np.concatenate([q[i] for q in tmp]) for i in (0, 1)]
    if full:
        return mse, cost
    else:
        return mse.mean()


def objective_cost(system, k, ABCLhat, delay=1, full=False, update_current=False, train=False):
    """Run 10000 episodes using estimates for matrices K and A, B, C, L.

    Parameters
    ----------
    system : System
        instance of class System
    k : ndarray, shape (k,m)
        The estimate for matrix K
    ABCLhat : tuple of ndarrays
        The estimates for matrices A, B, C, and L.
    delay : int
        delay >= 1
    full : bool
        True:  full output: MSEs, costs
        False: output mean of the cost
    update_current : bool
        True:  Action U[t] is based on Xhat[t], which is updated using past
               prediction error (Y[t-delay]-C Xhat[t-delay])
        False: Action U[t] is based on Xpred[t], which is based on Xhat[t+1-delay]
    train : bool
        training (True) or validation (False)

    Returns
    -------
    see argument 'full'
    """
    Chat = ABCLhat[2]

    samples = 10000
    splits = 10
    init_seed = 10000 if train else 0

    def foo(seed, samples=samples//splits):
        cost = np.zeros(samples)
        mse = np.zeros(samples)
        for i, seed in enumerate(range(seed, seed+samples)):
            if update_current:
                U, X, Y, _, _, Xhat, _ = system.LQG(
                    seed, ABCLhat=ABCLhat, delay=delay,
                    actor=('-np.array(' + str(k.tolist()) + ').dot(Xhat[t])'))
                mse[i] = np.mean((Y-Xhat.dot(Chat.T))**2)
            else:
                U, X, Y, _, _, _, Xpred = system.LQG(
                    seed, ABCLhat=ABCLhat, delay=delay, update_current=False,
                    actor=('-np.array(' + str(k.tolist()) + ').dot(Xpred[t])'))
                mse[i] = np.mean((Y-Xpred.dot(Chat.T))**2)
            cost[i] = np.trace(X.T.dot(X).dot(system.Q)) + np.trace(U.T.dot(U).dot(system.R))
        return mse, cost
    tmp = parmap(foo, range(init_seed, init_seed+samples, samples//splits))
    mse, cost = [np.concatenate([q[i] for q in tmp]) for i in (0, 1)]
    if full:
        return mse, cost
    else:
        return cost.mean()


def objective_modelfree(system, k, delay=1, train=False):
    """Run 10000 episodes using model-free agent with control matrix K.

    Parameters
    ----------
    system : System
        instance of class System
    k : ndarray, shape (k,m)
        The estimate for matrix K
    delay : int
        delay >= 1
    train : bool
        training (True) or validation (False)

    Returns
    -------
    costs : ndarray, shape (samples,)
    """
    samples = 10000
    splits = 10
    init_seed = 10000 if train else 0

    def foo(seed, samples=samples//splits):
        cost = np.zeros(samples)
        for i, seed in enumerate(range(seed, seed+samples)):
            U, X, Y, _, _ = system.LQG(
                seed, delay=delay,
                actor=('-np.array(' + str(k.tolist()) + ').dot(Y[t-delay] * (t>=delay))'))
            cost[i] = np.trace(X.T.dot(X).dot(system.Q)) + np.trace(U.T.dot(U).dot(system.R))
        return cost
    cost = parmap(foo, range(init_seed, init_seed+samples, samples//splits))
    return np.concatenate(cost)


if __name__ == "__main__":
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

    max_delay = 11  # max delay+1

    for ss in (0, 1):
        try:
            results = np.load('results/LQG_delay' + ('_C=I.npy', '.npy')
                              [ss], allow_pickle=True).item()
        except FileNotFoundError:
            s = (s0, s1)[ss]
            results = {'kalman': {'performance': np.zeros((2, max_delay, 10000))}}
            results['kalman']['performance'][:, 0] = objective_kalman(s)
            for delay in range(1, max_delay):
                results['kalman']['performance'][:, delay] = objective(
                    s, (s.A, s.B, s.C, np.array([s.A.dot(l) for l in s.L])), delay, full=True)

            # use different seeds for training and evaluation to avoid overfitting
            for i, j in ((False, 'past'), (True, 'current'), (False, 'past2'), (True, 'current2')):
                results[j] = {'performance': np.nan * np.zeros((2, max_delay, 10000)),
                              'Lhat': np.nan * np.zeros((max_delay, s.m, s.n)),
                              'Khat': np.nan * np.zeros((max_delay, s.B.shape[1], s.m))}
                for delay in range(1, max_delay):
                    if j[-1] == '2':
                        # optimize L by minimizing MSE, K by minimizing cost
                        res = minimize(lambda a: objective(
                            s, (s.A, s.B, s.C, a.reshape(s.m, s.n)), delay=delay, update_current=i,
                            train=True, actor='.5*np.random.randn(1)'), s.A.dot(s.L[-1]))
                        Lhat = res.x.reshape(s.m, s.n)
                        res = minimize(lambda a: objective_cost(
                            s, a, (s.A, s.B, s.C, Lhat), delay=delay, update_current=i, train=True),
                            s.K[0])
                        Khat = res.x.reshape(s.B.shape[1], s.m)
                    else:
                        # optimize L and K by minimizing cost
                        res = minimize(lambda a: objective_cost(
                            s, a[s.m*s.n:], (s.A, s.B, s.C, a[:s.m*s.n].reshape(s.m, s.n)),
                            delay=delay, update_current=i, train=True),
                            np.concatenate([s.A.dot(s.L[-1]).ravel(), s.K[0].ravel()]))
                        Lhat = res.x[:s.m*s.n].reshape(s.m, s.n)
                        Khat = res.x[s.m*s.n:].reshape(s.B.shape[1], s.m)
                    results[j]['Lhat'][delay] = Lhat
                    results[j]['Khat'][delay] = Khat
                    results[j]['performance'][:, delay] = objective_cost(
                        s, Khat, (s.A, s.B, s.C, Lhat), delay=delay, update_current=i, full=True)

            results['model-free'] = {'performance': np.nan * np.zeros((2, max_delay, 10000)),
                                     'Khat': np.zeros((max_delay, s.B.shape[1], s.n))}
            for delay in range(max_delay):
                res = minimize(lambda a: objective_modelfree(s, a, delay=delay, train=True).mean(),
                               np.zeros((s.B.shape[1], s.n)))
                Khat = res.x.reshape(s.B.shape[1], s.n)
                results['model-free']['Khat'][delay] = Khat
                results['model-free']['performance'][1, delay] = objective_modelfree(
                    s, Khat, delay=delay)

            np.save('results/LQG_delay' + ('_C=I.npy', '.npy')[ss], results)

        plt.figure(figsize=(6.4, 4.8))
        ax = plt.subplot(111)
        plt.xlabel('Delay')
        plt.ylabel('Cost')
        ax2 = inset_axes(ax, width=2.8, height=2.6, bbox_to_anchor=(.99, .75),
                         bbox_transform=ax.transAxes)
        for k, a in enumerate((ax, ax2)):
            for i, j in enumerate(('kalman', 'past', 'current', 'model-free')):
                a.errorbar(range(1-k, max_delay), results[j]['performance'][1].mean(1)[1-k:],
                           results[j]['performance'][1].std(1)[1-k:]/np.sqrt(9999),
                           c='C%g' % i, label=('LQG', 'ANN', 'Bio-OFC', 'Model-free')[i])
            mi = results['kalman']['performance'][1].mean(1)[1-k]
            tmp = results[('past', 'model-free')[k]]['performance'][1, 1:]
            ma = max(tmp.mean(1) + tmp.std(1)/np.sqrt(9999))
            a.set_ylim(mi - .02*(ma-mi), ma + .02*(ma-mi))
        ax.legend(loc=2)
        ax2.set_yticks([5, 10, 15])
        plt.tight_layout(pad=.05)
        plt.savefig('fig/LQG_delay' + ('_C=I.pdf', '.pdf')[ss])
