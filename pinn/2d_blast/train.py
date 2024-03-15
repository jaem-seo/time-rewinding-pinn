#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mycallbacks import MyModelCheckpoint, PlotResult
import pandas as pd

# Set random seed
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)
dde.backend.tf.random.set_random_seed(seed)
    
# Set hyperparameters
n_output = 4 # rho, vx, vy, energy

num_domain = 10000
num_boundary = 1000
num_x = 100

Lx = 4.0
Ly = 4.0
x0, y0 = 2.75, 2.75

n_adam = 500
n_lbfgs = 1000
n_repeat = 1

lr = 1e-2 # for Adam
loss_weights = [1., 1., 1., 1.] + [1., 1., 1., 1.] + [1., 1., 1., 1.]

# Set physical parameters
xmin, xmax = 0., Lx
ymin, ymax = 0., Ly
tmin, tmax, tobs = 0.1, 1.0, 1.0

gamma = 1.4
rho0 = 1.0
p0 = 0.000005
v0 = 0

obs_path = '2d_solution.csv'


def plot_history(model):
    steps = model.losshistory.steps
    losses = np.array(model.losshistory.loss_train)
    total_loss = np.sum(losses, axis=1)
    pde_loss = np.sum(losses[:, :3], axis=1)
    bc_loss = np.sum(losses[:, 3:6], axis=1)
    ic_loss = np.sum(losses[:, 6:9], axis=1)
    
    plt.figure(figsize=(6, 2))
    plt.semilogy(steps, total_loss, 'k', label='total loss')
    #plt.semilogy(steps, pde_loss, 'b', label='pde loss')
    #plt.semilogy(steps, bc_loss, 'r', label='bc loss')
    #plt.semilogy(steps, ic_loss, 'g', label='ic loss')
    plt.legend()
    plt.savefig('history.svg')
    plt.show()

def pde(x, u):
    rho, vx, vy, eng = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4]
    rvx, rvy = rho * vx, rho * vy
    p = (gamma - 1) * (eng - 0.5 * (rvx * vx + rvy * vy))
    ft = [rho, rvx, rvy, eng]
    fx = [rvx, rvx * vx + p, rvx * vy, vx * (eng + p)]
    fy = [rvy, rvy * vx, rvy * vy + p, vy * (eng + p)]
    
    pdes = []
    for i in range(n_output):
        ft_t = dde.grad.jacobian(ft[i], x, i=0, j=2)
        fx_x = dde.grad.jacobian(fx[i], x, i=0, j=0)
        fy_y = dde.grad.jacobian(fy[i], x, i=0, j=1)
        pdes.append(ft_t + fx_x + fy_y)
    
    # incompressible
    #vx_x = dde.grad.jacobian(vx, x, i=0, j=0)
    #vy_y = dde.grad.jacobian(vy, x, i=0, j=1)
    #pdes.append(vx_x + vy_y)
    
    return pdes

def obs_data():
    df = pd.read_csv(obs_path)
    r, rho, vr, p = df['r'].values, df['rho'].values, df['v'].values, df['p'].values
    theta = np.linspace(0, 2 * np.pi, 101)
    R, Theta = np.meshgrid(r, theta)
    Rho, _ = np.meshgrid(rho, theta)
    P, _ = np.meshgrid(p, theta)
    Vr, _ = np.meshgrid(vr, theta)
    Vx, Vy = Vr * np.cos(Theta), Vr * np.sin(Theta)
    R_fluc = R - 0.15 * np.cos(5 * Theta)
    X, Y = x0 + R_fluc * np.cos(Theta), y0 + R_fluc * np.sin(Theta)
    
    xyt = np.vstack((X.flatten(), Y.flatten(), np.full(np.size(X), tobs))).T
    u = np.zeros((len(xyt), n_output))
    u[:, 0] = Rho.flatten()
    u[:, 1] = Rho.flatten() * Vx.flatten()
    u[:, 2] = Rho.flatten() * Vy.flatten()
    u[:, 3] = P.flatten() / (gamma - 1) + 0.5 * Rho.flatten() * (Vx.flatten() ** 2 + Vy.flatten() ** 2)
    idx = (xyt[:, 0] >= 0) * (xyt[:, 0] <= Lx) * (xyt[:, 1] >= 0) * (xyt[:, 1] <= Ly)
    
    return xyt[idx], u[idx]

def boundary(xyt, on_boundary):
    return on_boundary

def bc_u0(xy):
    return rho0 * np.ones_like(xy[:, [0]])

def bc_u1(xy):
    return np.zeros_like(xy[:, [0]])

def bc_u2(xy):
    return np.zeros_like(xy[:, [0]])

def bc_u3(xy):
    return np.zeros_like(xy[:, [0]]) + p0 / (gamma - 1)


# Set domain
geom = dde.geometry.Rectangle([0., 0.], [Lx, Ly])
timedomain = dde.geometry.TimeDomain(tmin, tmax)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
        
# Set initial & boundary conditions
xy0, u0 = obs_data()

func_bcs = [bc_u0, bc_u1, bc_u2, bc_u3]
bcs, ics = [], []
for i in range(n_output):
    ics.append(dde.icbc.PointSetBC(xy0, u0[:, [i]], component=i))
    bcs.append(dde.icbc.DirichletBC(geomtime, func_bcs[i], boundary, component=i))


if __name__ == "__main__":    
    # Build loss and model
    net = dde.nn.FNN([3] + [64] * 4 + [n_output], ["tanh", "tanh", "tanh", "tanh", None], "Glorot normal")
    dde.optimizers.config.set_LBFGS_options(\
        #ftol=np.nan, gtol=np.nan, maxiter=n_lbfgs, maxfun=n_lbfgs,
    )
    
    resampler = dde.callbacks.PDEPointResampler(period=10)
    ckpt = MyModelCheckpoint("best_model", save_better_only=True, start_step=1000, verbose=1)
    plot_callback = PlotResult(xmin, xmax, ymin, ymax, tmin, tmax, period=10000)
    
    data = dde.data.TimePDE(
        geomtime, 
        pde, 
        bcs + ics,
        num_domain=num_domain, 
        num_boundary=num_boundary, 
    )
    model = dde.Model(data, net)
    
    # Compile and train
    for _ in range(n_repeat):
        if n_adam > 0:
            model.compile("adam", lr=lr, loss_weights=loss_weights)
            losshistory, train_state = model.train(iterations=n_adam, display_every=100, callbacks=[resampler, ckpt, plot_callback])
        
        if n_lbfgs > 0:
            model.compile("L-BFGS", loss_weights=loss_weights)
            losshistory, train_state = model.train(display_every=100, callbacks=[resampler, ckpt, plot_callback])
        
        #dde.saveplot(losshistory, train_state, issave=False, isplot=True)
    
        
    try:
        dde.saveplot(losshistory, train_state, issave=True, isplot=True, loss_fname=f"loss{seed}.dat", train_fname=f"train{seed}.dat", test_fname=f"test{seed}.dat")
    except:
        plot_history(model)
    
    model.save('saved_model')
    #model.save('saved_model', protocol='pickle')
    #model.restore("best_model.ckpt")
    
    xtmp, ytmp = np.linspace(xmin, xmax, num_x), np.linspace(ymin, ymax, num_x)
    Xtmp, Ytmp = np.meshgrid(xtmp, ytmp)
    
    # Plot results
    times = np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * (tmax - tmin) + tmin
    cs = ['b', 'r', 'g']
    xs = [np.zeros([num_x ** 2, 3]) for i in range(len(times))]
    for i in range(len(times)):
        xs[i][:, 0] = Xtmp.flatten()
        xs[i][:, 1] = Ytmp.flatten()
        xs[i][:, 2] = times[i]
    
    ys = [model.predict(x) for x in xs]
    
    
    fig, axs = plt.subplots(4, len(times), figsize=(12, 12), sharex=True, sharey=True)
    lbs = ['rho', 'vx', 'vy', 'p']
    
    for i in range(len(times)):
        for j in range(n_output):
            cs = axs[j, i].contourf(Xtmp, Ytmp, ys[i][:, j].reshape([num_x, num_x]))
            if j == 0:
                axs[j, i].set_title(f't={times[i]:.3f}')
                
            fig.colorbar(cs, ax=axs[j, i])
    
    plt.show()
