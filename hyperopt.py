from ofc import System, parmap
import numpy as np
import optuna
from sys import argv


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
        try:
            if loop == 'closed':
                result = system.PGwithSysID(
                    seed, eta, etaK, momentum, sigma, delay, episodes,
                    ABCLhat=(Ahat, Bhat, Chat, Lhat), x0=x0)
                result = np.mean(result[0]) + np.mean(result[1])
            else:
                sysIdresult = system.SysID(Ahat, Bhat, Chat, Lhat, eta, delay,
                                           episodes//2, x0=x0, init_seed=seed)
                if np.isnan(sysIdresult[4]).sum() > 0:
                    result = 1000  # np.inf
                else:
                    result = system.PGafterSysID(seed, etaK, momentum, sigma, delay,
                                                 episodes//2, x0=x0, ABCLhat=sysIdresult[:4])
                    result = np.mean(result[0]) + np.mean(result[1])
        except:
            result = 1000  # np.inf
        return result
    return np.mean(parmap(foo, range(runs)))


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


ss = int(argv[1])
delay = int(argv[2])
loop = argv[3]
sigma = float(argv[4])
momentum = float(argv[5])
s = (s0, s1, s2)[ss]

if momentum == 0:
    def opt(trial):
        eta = trial.suggest_float('eta', 5e-4, 2e-2, log=True)
        if sigma in (.1, .05):
            etaK = trial.suggest_float('etaK', 5e-4, 5e-3, log=True)
        elif sigma == .2:
            etaK = trial.suggest_float('etaK', 1e-4, 1e-3, log=True)
        elif sigma == .5:
            etaK = trial.suggest_float('etaK', 5e-6, 1e-4, log=True)
        return run(s, eta, etaK, momentum, sigma, delay=delay, loop=loop)
elif momentum == .9:
    def opt(trial):
        eta = trial.suggest_float('eta', 5e-4, 2e-2, log=True)
        if sigma in (.1, .05):
            etaK = trial.suggest_float('etaK', 7e-5, 7e-4, log=True)
        elif sigma == .2:
            etaK = trial.suggest_float('etaK', 2e-5, 2e-4, log=True)
        elif sigma == .5:
            etaK = trial.suggest_float('etaK', 5e-7, 1e-5, log=True)
        return run(s, eta, etaK, momentum, sigma, delay=delay, loop=loop)
elif momentum == .99:
    def opt(trial):
        eta = trial.suggest_float('eta', 1e-3, 1e-2, log=True)
        if sigma in (.1, .05):
            etaK = trial.suggest_float('etaK', 1e-5, 1e-4, log=True)
        elif sigma == .2:
            etaK = trial.suggest_float('etaK', 3e-6, 3e-5, log=True)
        elif sigma == .5:
            etaK = trial.suggest_float('etaK', 1e-7, 1e-6, log=True)
        return run(s, eta, etaK, momentum, sigma, delay=delay, loop=loop)
elif momentum == .9995:
    def opt(trial):
        if loop == 'closed':
            eta = trial.suggest_float('eta', 2e-3, 2e-2, log=True)
        else:
            eta = trial.suggest_float('eta', 1e-3, 1e-2, log=True)
        if sigma == .05:
            etaK = trial.suggest_float('etaK', 1e-6, 3e-5, log=True)
        elif sigma == .1:
            etaK = trial.suggest_float('etaK', 1e-6, 1e-5, log=True)
        elif sigma == .2:
            etaK = trial.suggest_float('etaK', 2e-7, 2e-6, log=True)
        elif sigma == .5:
            etaK = trial.suggest_float('etaK', 1e-8, 1e-7, log=True)
        return run(s, eta, etaK, momentum, sigma, delay=delay, loop=loop)

study = optuna.create_study()  # Create a new study.
study.optimize(opt, n_trials=1000)  # Invoke optimization of the objective function.
np.save('results/LQG_%s_delay%g_sigma%g_momentum%g%s' %
        (loop, delay, sigma, momentum, ('_C=I.npy', '.npy', '_m=3.npy')[ss]), study)
