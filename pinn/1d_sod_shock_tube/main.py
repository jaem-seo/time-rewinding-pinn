#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sodshock
import pickle

# Set random seed
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)
dde.backend.tf.random.set_random_seed(seed)
    
# Set hyperparameters
n_output = 3 # rho, u, energy
num_domain = 1000
num_boundary = 100
num_initial = 1000

n_adam = 0
n_lbfgs = 3000
n_repeat = 1

lr = 1e-3 # for Adam
loss_weights = [1., 1., 1.] + [1., 1., 1.] + [100., 100., 100.]

# Set physical parameters
xmin, xmax = 0., 1.
tmin, tmax, tobs = -0.1, 0.1, 0.
gamma = 1.4

def pde(x, u):
    rho, vx, eng = u[:, 0:1], u[:, 1:2], u[:, 2:3]
    rvx = rho * vx
    p = (gamma - 1) * (eng - 0.5 * rvx * vx)
    ft = [rho, rvx, eng]
    fx = [rvx, rvx * vx + p, vx * (eng + p)]
    
    pdes = []
    for i in range(n_output):
        ft_t = dde.grad.jacobian(ft[i], x, i=0, j=1)
        fx_x = dde.grad.jacobian(fx[i], x, i=0, j=0)
        pdes.append(ft_t + fx_x)
    
    return pdes

positions, regions, values = sodshock.solve(left_state=(1., 1., 0.), \
    right_state=(0.1, 0.125, 0.), geometry=(0., 1., 0.5), t=0.1, 
    gamma=gamma, npts=num_initial, dustFrac=0.)

def ic_func(x):
    ic = np.zeros((len(x), n_output))
    ic[:, 0] = values['rho']
    ic[:, 1] = values['u']
    ic[:, 2] = values['p'] / (gamma - 1) + 0.5 * values['rho'] * values['u'] ** 2
    return ic

def boundary(x, on_boundary):
    return on_boundary

def bc_u0(x):
    return 1.0 * (x[:, [0]] <= 0.5) + 0.125 * (x[:, [0]] > 0.5)

def bc_u1(x):
    return 0.0 * x[:, [0]]

def bc_u2(x):
    return 2.5 * (x[:, [0]] <= 0.5) + 0.25 * (x[:, [0]] > 0.5)


# Set domain
geom = dde.geometry.Interval(xmin, xmax)
timedomain = dde.geometry.TimeDomain(tmin, tmax)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Set initial & boundary conditions
x0 = np.vstack((np.linspace(geom.l, geom.r, num=num_initial), np.full((num_initial), tobs))).T
y0 = ic_func(x0)

func_bcs = [bc_u0, bc_u1, bc_u2]
bcs, ics = [], []
for i in range(n_output):
    ics.append(dde.icbc.PointSetBC(x0, y0[:, [i]], component=i))
    bcs.append(dde.icbc.DirichletBC(geom, func_bcs[i], boundary, component=i))


if __name__ == "__main__":    
    # Build loss and model
    data = dde.data.TimePDE(
        geomtime, 
        pde, 
        bcs + ics, 
        num_domain=num_domain, 
        num_boundary=num_boundary, 
    )
    net = dde.nn.FNN([2] + [32] * 3 + [n_output], ["tanh", "tanh", "tanh", None], "Glorot normal")#, "l2")
    model = dde.Model(data, net)
    
    # Compile and train
    resampler = dde.callbacks.PDEPointResampler(period=10)
    for _ in range(n_repeat):
        if n_adam > 0:
            model.compile("adam", lr=lr, loss_weights=loss_weights)
            losshistory, train_state = model.train(iterations=n_adam, display_every=10, callbacks=[resampler])
        
        if n_lbfgs > 0:
            dde.optimizers.config.set_LBFGS_options(
                maxcor=200,
                ftol=np.nan,
                gtol=np.nan,
                maxiter=n_lbfgs,
                maxfun=n_lbfgs,
                maxls=100,
            )
            model.compile("L-BFGS", loss_weights=loss_weights)
            losshistory, train_state = model.train(display_every=10, callbacks=[resampler])
        #dde.saveplot(losshistory, train_state, issave=False, isplot=True)
    
    dde.saveplot(losshistory, train_state, issave=True, isplot=True, loss_fname=f"loss{seed}.dat", train_fname=f"train{seed}.dat", test_fname=f"test{seed}.dat")
    model.save('saved_model')
    model.save('saved_model', protocol='pickle')
    
    
    # Plot results
    times = np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * (tmax - tmin) + tmin
    xref = np.linspace(0, 1, 51)
    cs = ['b', 'r', 'g']
    xs = [np.zeros([len(xref), 2]) for i in range(len(times))]
    for i in range(len(times)):
        xs[i][:, 0] = xref
        xs[i][:, 1] = times[i]
        
    fig, axs = plt.subplots(1, len(times), figsize=(12, 2), sharey=True)
    lbs = ['rho', 'u', 'eng']
    ys = [model.predict(x) for x in xs]
    
    axs[0].set_ylabel(f'u (seed={seed})')
    for i in range(len(times)):
        u = np.zeros_like(ys[i])
        positions, regions, values = sodshock.solve(left_state=(1., 1., 0.), \
            right_state=(0.1, 0.125, 0.), geometry=(0., 1., 0.5), t=0.1 + times[i], 
            gamma=gamma, npts=51, dustFrac=0.)
        for j in range(n_output):
            axs[i].plot(xref, ys[i][:, j], cs[j])
            if lbs[j] == 'eng':
                axs[i].plot(xref, values['p'] / (gamma - 1) + 0.5 * values['rho'] * values['u'] ** 2, 'k--')
            else:
                axs[i].plot(xref, values[lbs[j]], 'k--')
        axs[i].set_title(f't={times[i]:.3f}')
        axs[i].set_xlabel('x')
    
    plt.show()
    
    
    fig, axs = plt.subplots(1, len(times), figsize=(12, 2), sharey=True)
    lbs = ['rho', 'u', 'p']
    ys = [model.predict(x) for x in xs]
    
    axs[0].set_ylabel(f'u (seed={seed})')
    for i in range(len(times)):
        u = np.zeros_like(ys[i])
        u[:, 0] = ys[i][:, 0]
        u[:, 1] = ys[i][:, 1]
        u[:, 2] = (gamma - 1) * (ys[i][:, 2] - 0.5 * ys[i][:, 0] * u[:, 1] ** 2)
        positions, regions, values = sodshock.solve(left_state=(1., 1., 0.), \
            right_state=(0.1, 0.125, 0.), geometry=(0., 1., 0.5), t=0.1 + times[i], 
            gamma=gamma, npts=51, dustFrac=0.)
        for j in range(n_output):
            axs[i].plot(xref, u[:, j], cs[j])
            axs[i].plot(xref, values[lbs[j]], 'k--')
        axs[i].set_title(f't={times[i]:.3f}')
        axs[i].set_xlabel('x')
    
    plt.show()