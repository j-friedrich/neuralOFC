""" Script to produce Figs. 2 and S4 """

from fig3_delay import objective
from ofc import System, parmap
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# colors from Okabe & Ito's colorblind friendly palette
colors = ["#0072B2", "#009E73", "#D55E00", "#E69F00"]
plt.rc('axes', prop_cycle=plt.cycler('color', colors))
plt.rc('font', size=18)
plt.rc('legend', **{'fontsize': 12})


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
            C=np.eye(2),
            V=.01 * np.eye(2),
            W=np.diag([.04, .01]),
            Q=np.array([[1, 0], [0, 0]]),
            R=np.ones((1, 1)),
            T=11)

s2 = System(A=np.array([[1, 1], [0, 1]]),
            B=np.array([[0], [1]]),
            C=np.eye(2),
            V=.01 * np.eye(2),
            W=np.diag([.01, .01]),
            Q=np.array([[1, 0], [0, 0]]),
            R=np.ones((1, 1)),
            T=11)

try:
    optLs = [np.load('results/LQG_delay_C=I.npy',
                    allow_pickle=True).item()['current2']['Lhat'][1]]
except FileNotFoundError:
    print("Output file of fig3_delay.py not found. Run fig3_delay.py first.")
for ss in (0, 1):
    s = (s1, s2)[ss]
    res = minimize(lambda a: objective(
        s, (s.A, s.B, s.C, a.reshape(s.m, s.n)), delay=1, update_current=True,
        train=True, actor='.5*np.random.randn(1)'), s.A.dot(s.L[-1]))
    optLs.append(res.x.reshape(s.m, s.n))
optLs.append(optLs[0])

optmse = []
for s, l in zip((s0, s1, s2), optLs[:3]):
    optmse.append(objective(s, (s.A, s.B, s.C, l), actor='.5*np.random.randn(1)'))
optmse.append(optmse[0])


episodes = 2500


def foo(lr, seed=0, useL=True):
    eta = (0, 0, 0, lr)
    res0 = s0.SysID(s0.A, s0.B, s0.C, optLs[0], eta, init_seed=seed, useL=useL, verbose=True)
    res1 = s1.SysID(*res0[:4], eta, init_seed=seed+episodes, useL=useL, verbose=True)
    res2 = s2.SysID(*res1[:4], eta, init_seed=seed+2*episodes, useL=useL, verbose=True)
    res3 = s0.SysID(*res2[:4], eta, init_seed=seed+3*episodes, useL=useL, verbose=True)
    mse = np.concatenate([res0[4], res1[4], res2[4], res3[4]])
    Ls = np.concatenate([res0[5], res1[5], res2[5], res3[5]])
    return mse, Ls


# use local learning rule 
opt = minimize(lambda lr: np.mean(
    parmap(lambda seed: foo(lr, seed*4*episodes)[0], range(20))), .05)
tmp = parmap(lambda seed: foo(opt.x, seed*4*episodes), range(20))
mse = np.array([j[0] for j in tmp])
Ls = np.array([j[1] for j in tmp])

plt.figure(figsize=(6, 4))
plt.plot(Ls.mean(0).reshape(-1, 4), alpha=.5, lw=1.5)
for i, l in enumerate(optLs):
    for k in range(4):
        plt.plot([i*episodes, (i+1)*episodes], [l.ravel()[k]]*2,
                 '--', c='C%g' % k, zorder=-11, lw=2.5)
plt.xlabel('Episodes')
plt.ylabel('Filter Gain')
plt.tight_layout(pad=.05)
plt.savefig('fig/varying_noise_gain_C=I.pdf')

plt.figure(figsize=(6, 4))
plt.plot(mse.mean(0), alpha=.5, lw=.5)
for i, l in enumerate(optmse):
    plt.plot([i*episodes, (i+1)*episodes], [l]*2, '--', c='C0', lw=2.5)
plt.xlabel('Episodes')
plt.ylabel('MSE')
plt.tight_layout(pad=.05)
plt.savefig('fig/varying_noise_mse_C=I.pdf')


# use non-local SGD learning rule 
opt_C = minimize(lambda lr: np.mean(
    parmap(lambda seed: foo(lr, seed*4*episodes, False)[0], range(20))), .01)
tmp_C = parmap(lambda seed: foo(opt_C.x, seed*4*episodes, False), range(20))
mse_C = np.array([j[0] for j in tmp_C])
Ls_C = np.array([j[1] for j in tmp_C])

plt.figure(figsize=(6, 4))
plt.plot(Ls_C.mean(0).reshape(-1, 4), alpha=.5, lw=1.5)
for i, l in enumerate(optLs):
    for k in range(4):
        plt.plot([i*episodes, (i+1)*episodes], [l.ravel()[k]]*2,
                 '--', c='C%g' % k, zorder=-11, lw=2.5)
plt.xlabel('Episodes')
plt.ylabel('Filter Gain')
plt.tight_layout(pad=.05)
plt.savefig('fig/varying_noise_gain_C=I_useC.pdf')

plt.figure(figsize=(6, 4))
plt.plot(mse_C.mean(0), alpha=.5, lw=.5)
for i, l in enumerate(optmse):
    plt.plot([i*episodes, (i+1)*episodes], [l]*2, '--', c='C0', lw=2.5)
plt.xlabel('Episodes')
plt.ylabel('MSE')
plt.tight_layout(pad=.05)
plt.savefig('fig/varying_noise_mse_C=I_useC.pdf')

print('Objective local learning rule: %.4f' % opt.fun)
print('Objective non-local SGD  rule: %.4f' % opt_C.fun)