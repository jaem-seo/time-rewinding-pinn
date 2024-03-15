#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sodshock
import pickle
from mycallbacks import MyModelCheckpoint, PlotResult

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

Lx = 0.5
Ly = 1.5

n_adam = 200000
n_lbfgs = 1000
n_repeat = 1

lr = 1e-3 # for Adam
loss_weights = [1., 1., 1., 1.] + [1., 1., 1., 1.] + [1., 1., 1., 1.] + [10., 10., 10., 10.]
#loss_weights = [10., 10., 10., 10., 1.] + [1., 1., 1., 1.] + [1., 1., 1., 1.] + [10., 10., 10., 10.]

# Set physical parameters
xmin, xmax = 0., Lx
ymin, ymax = 0., Ly
tmin, tmax, tobs = 0.0, 10., 10.
gamma = 1.4
#g = -0.1
g = -0.1
rhob, rhot = 1., 2.
p0, pb, pt = 2.5, 2.575, 2.35

obs_path = 'data_10.00s.npz'

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
    #plt.show()

def pde(x, u):
    rho, vx, vy, eng = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4]
    rvx, rvy = rho * vx, rho * vy
    p = (gamma - 1) * (eng - 0.5 * (rvx * vx + rvy * vy))
    ft = [rho, rvx, rvy, eng]
    fx = [rvx, rvx * vx + p, rvx * vy, vx * (eng + p)]
    fy = [rvy, rvy * vx, rvy * vy + p, vy * (eng + p)]
    s = [0.0, 0.0, rho * g, rvy * g]
    
    pdes = []
    for i in range(n_output):
        ft_t = dde.grad.jacobian(ft[i], x, i=0, j=2)
        fx_x = dde.grad.jacobian(fx[i], x, i=0, j=0)
        fy_y = dde.grad.jacobian(fy[i], x, i=0, j=1)
        pdes.append(ft_t + fx_x + fy_y - s[i])
    
    # incompressible
    #vx_x = dde.grad.jacobian(vx, x, i=0, j=0)
    #vy_y = dde.grad.jacobian(vy, x, i=0, j=1)
    #pdes.append(vx_x + vy_y)
    
    return pdes

def obs_data(obs_path, tobs):
    obs = np.load(obs_path)
    rho, vx, vy, p = obs['rho'], obs['vx'], obs['vy'], obs['p']
    #rho, vx, vy, p = np.round(obs['rho']), obs['vx'], obs['vy'], obs['p']
    rho = 0.5 * (rho + rho[::-1])
    vx = 0.5 * (vx + vx[::-1])
    vy = 0.5 * (vy + vy[::-1])
    p = 0.5 * (p + p[::-1])
    nx, ny = rho.shape
    dx = Lx / nx
    x = np.linspace(xmin + 0.5 * dx, xmax - 0.5 * dx, nx)
    y = np.linspace(ymin + 0.5 * dx, ymax - 0.5 * dx, ny)
    Y, X = np.meshgrid(y, x)
    
    xyt = np.vstack((X.flatten(), Y.flatten(), np.full(np.size(X), tobs))).T
    u = np.zeros((len(xyt), n_output))
    u[:, 0] = rho.flatten()
    u[:, 1] = vx.flatten()
    u[:, 2] = vy.flatten()
    u[:, 3] = p.flatten() / (gamma - 1) + 0.5 * rho.flatten() * (vx.flatten() ** 2 + vy.flatten() ** 2)
    
    return xyt, u

def boundary(xyt, on_boundary):
    return on_boundary

def boundary_xonly(xyt, on_boundary):
    return on_boundary and (xyt[1] != ymin) and (xyt[1] != ymax)

def boundary_yonly(xyt, on_boundary):
    return on_boundary and (xyt[0] != xmin) and (xyt[0] != xmax)
    #return on_boundary and (xyt[0] != xmin) and (xyt[0] != xmax) and np.isclose(xyt[1], ymin)

def bc_u0(xy):
    return rhob * (xy[:, [1]] < 0.5 * Ly) + rhot * (xy[:, [1]] >= 0.5 * Ly)

def bc_u1(xy):
    return 0.0 * xy[:, [0]]

def bc_u2(xy):
    return 0.0 * xy[:, [0]]

def bc_u3(xy):
    x, y = xy[:, [0]], xy[:, [1]]
    #p = p0 + g * (y-0.75) * rho
    p = pb * (y < 0.5 * Ly) + pt * (y >= 0.5 * Ly)
    return p / (gamma - 1)


# Set domain
geom = dde.geometry.Rectangle([0., 0.], [Lx, Ly])
timedomain = dde.geometry.TimeDomain(tmin, tmax)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
        
# Set initial & boundary conditions
xy0, u0 = obs_data(obs_path, tobs)

func_bcs = [bc_u0, bc_u1, bc_u2, bc_u3]
bcxs, bcys, ics = [], [], []
for i in range(n_output):
    ics.append(dde.icbc.PointSetBC(xy0, u0[:, [i]], component=i))
    bcxs.append(dde.icbc.PeriodicBC(geomtime, 0, boundary_xonly, component=i))
    bcys.append(dde.icbc.DirichletBC(geomtime, func_bcs[i], boundary_yonly, component=i))


if __name__ == "__main__":    
    # Build loss and model
    net = dde.nn.FNN([3] + [64] * 4 + [n_output], ["tanh", "tanh", "tanh", "tanh", None], "Glorot normal")
    dde.optimizers.config.set_LBFGS_options(\
        #ftol=np.nan, gtol=np.nan, maxiter=n_lbfgs, maxfun=n_lbfgs,
    )
    
    resampler = dde.callbacks.PDEPointResampler(period=10)
    ckpt = MyModelCheckpoint("best_model", save_better_only=True, start_step=50000, verbose=1)
    plot_callback = PlotResult(xmin, xmax, ymin, ymax, tmin, tmax, period=20000)
    
    data = dde.data.TimePDE(
        geomtime, 
        pde, 
        bcxs + bcys + ics,
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
            losshistory, train_state = model.train(display_every=100, callbacks=[resampler, ckpt])
        
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
    
    
    fig, axs = plt.subplots(1, len(times), figsize=(12, 4), sharex=True, sharey=True)
    
    for i in range(len(times)):
        cs = axs[i].contourf(Xtmp, Ytmp, ys[i][:, 0].reshape([num_x, num_x]))
        axs[i].set_title(f't={times[i]:.3f}')
        fig.colorbar(cs, ax=axs[i])
    
    plt.tight_layout()
    plt.savefig('evol.png')
    #plt.savefig('evol.svg')
    plt.show()
