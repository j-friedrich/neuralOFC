""" Script to produce Fig. S3 """

from ofc import System, parmap
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

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


for ss in (0, 1, 2):
    print('\nSystem', ss)
    s = (s0, s1, s2)[ss]
    eta = np.load('results/LQG_%s_delay%g_sigma%g_momentum%g%s' %
                        ('open', 1, .2, .99,
                        ('_C=I.npy', '.npy', '_m=3.npy')[ss]),
                        allow_pickle=True).item().best_params['eta']

    def foo(seed=0):
        np.random.seed(seed)
        if s.m == s.n:
            Ahat, Bhat = [.1*np.random.randn(*a_.shape)
                            for a_ in (s.A, s.B)]
            Chat, Lhat = [.5*np.eye(s.m) + .5*np.random.rand(*a.shape)
                            for a in (s.C, s.C.T)]
        else:
            Ahat, Bhat, Chat, Lhat = [.1*np.random.randn(*a_.shape) for a_ in
                                        (s.A, s.B, s.C, s.C.T)]
            while np.any(np.linalg.eigvals(Lhat.dot(Chat)) <= .01):
                Lhat = .1*np.random.randn(*Chat.T.shape)
        while np.any(np.linalg.eigvals(Lhat.dot(Chat)) <= .01):
            Lhat = .1*np.random.randn(*Chat.T.shape)
        return s.SysID(Ahat, Bhat, Chat, Lhat, eta, episodes=5000,
                        init_seed=seed, x0=[-1, 0, 0][:s.m], verbose=True)

    sysIdresult = parmap(foo, range(20))

    eCLe = np.array([j[-1] for j in sysIdresult])
    g1traj = eCLe.reshape(20, -1, 10).mean(-1)
    g10traj = eCLe.reshape(20, -1, 100).mean(-1)
    print('Percentage of cases where eCLe < 0')
    for l, g in ((' 1 step  ', eCLe), (' 1 epoch ', g1traj), ('10 epochs', g10traj)):
        print('%s %.3f%%' % (l, 100*(g < 0).sum()/g.size))
    plt.figure(figsize=(6, 4))
    plt.plot(g1traj.T)
    plt.xlabel('Episodes')
    plt.ylabel(r'$\sum_t^T e_t^\top C L e_t$')
    plt.ylim(-.2, 2.8)
    plt.tight_layout(pad=.05)
    plt.savefig('fig/eCLe' + ('_C=I.pdf', '.pdf', '_m=3.pdf')[ss])
