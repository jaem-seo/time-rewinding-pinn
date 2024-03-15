#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import sodshock

saved_path = "saved_model-20005.ckpt"

# Set hyperparameters
n_output = 3 # rho, u, energy
num_domain = 1000
num_boundary = 100
num_initial = 1000

n_adam = 0
n_lbfgs = 20000
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
    net = dde.nn.FNN([2] + [32] * 3 + [n_output], ["tanh", "tanh", "tanh", None], "Glorot normal", "l2")
    model = dde.Model(data, net)
    
    # Load model
    dde.optimizers.config.set_LBFGS_options(
        maxcor=200,
        ftol=np.nan,
        gtol=np.nan,
        maxiter=n_lbfgs,
        maxfun=n_lbfgs,
        maxls=100,
    )
    model.compile("L-BFGS")
    model.restore(saved_path)

    ylims = [[-0.05, 1.05], [-0.05, 1.05], [-0.05, 1.05], [1.7, 3.02]]

    # Plot loss history
    w = np.array([1, 1, 100])
    with open('loss0.dat', 'r') as f:
        a = f.readlines()[1:]

    step, loss_pde, loss_bc, loss_ic = [], [], [], []
    for line in a:
        step.append(float(line.split()[0]))
        loss_pde.append(sum(map(float, line.split()[1:4])))
        loss_bc.append(sum(map(float, line.split()[4:7])))
        loss_ic.append(sum(map(float, line.split()[7:10])))
    
    loss_sum = np.sum([loss_pde, loss_bc, loss_ic], axis=0)
    loss_pde = np.array(loss_pde) / w[0]
    loss_bc = np.array(loss_bc) / w[1]
    loss_ic = np.array(loss_ic) / w[2]

    fig, ax = plt.subplots(1, 1, figsize=(6, 2))
    ax.semilogy(step, loss_sum, 'k', lw=2, label='Weighted sum')
    ax.semilogy(step, loss_pde, 'b', lw=2, label='PDE loss')
    ax.semilogy(step, loss_bc, 'r', lw=2, label='BC loss')
    ax.semilogy(step, loss_ic, 'g', lw=2, label='IC loss')
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([-100, 3000])
    ax.set_ylim([None, 2e3])
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.legend(ncol=2, frameon=False)

    plt.tight_layout()
    #plt.savefig('loss_history.svg')
    plt.show()

    # Plot 3d profile of rho
    x = np.linspace(0, 1, 101)
    t = np.linspace(-0.1, 0.1, 101)
    X, T = np.meshgrid(x, t)
    xt = np.zeros([len(x) * len(t), 2])
    for i in range(len(t)):
        xt[i * len(x): (i+1) * len(x), 0] = x
        xt[i * len(x): (i+1) * len(x), 1] = t[i]

    y = model.predict(xt)
    Y = y.reshape(len(t), len(x), 3)

    fig = plt.figure(figsize=(3, 2.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d', computed_zorder=False)
    ax.plot_surface(X, T, Y[:, :, 0], cmap='viridis', linewidth=0, antialiased=False)
    #ax.plot_surface(X, T, Y[:, :, 0], linewidth=0, antialiased=False)
    ax.contour3D(X, T, Y[:, :, 0], 500, zorder=2)
    ax.plot(values['x'], np.zeros_like(values['x']), values['rho'], 'k--', lw=2, zorder=3)
    ax.set_zlim([-0.25, 1.25])
    ax.tick_params(pad=-1.5)
    ax.view_init(30, 330)

    #plt.savefig('3d_plot.svg')
    plt.show()

    # Plot initial states
    x, rho, vx, p = values['x'], values['rho'], values['u'], values['p']
    ei = p / (gamma - 1) / rho
    fig, axs = plt.subplots(4, 1, figsize=(2, 3), sharex=True)
    for i, y in enumerate([rho, vx, p, ei]):
        axs[i].plot(x, y, 'k', lw=2, )
        axs[i].set_xlim([0., 1.])
        axs[i].set_ylim(ylims[i])
        #axs[i].set_ylabel(' ')
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
    #axs[-1].set_xlabel(' ')

    plt.tight_layout()
    #plt.savefig('initial.svg')
    plt.show()
    

    # Plot PINN results
    times = np.array([-0.1, -0.05, 0.05, 0.1])
    #xref = np.linspace(0., 1., 101)
    xref = np.linspace(-0.5, 1.5, 101)
    #xref = np.linspace(-1., 2., 101)
    xs = [np.zeros([len(xref), 2]) for i in range(len(times))]
    for i in range(len(times)):
        xs[i][:, 0] = xref
        xs[i][:, 1] = times[i]
    
    ys = [model.predict(x) for x in xs]
    
    for i in range(len(times)):
        rho = ys[i][:, 0]
        vx = ys[i][:, 1]
        p = (gamma - 1) * (ys[i][:, 2] - 0.5 * rho * vx ** 2)
        ei = p / (gamma - 1) / rho
        
        positions, regions, values = sodshock.solve(left_state=(1., 1., 0.), \
            right_state=(0.1, 0.125, 0.), geometry=(xref[0], xref[-1], 0.5), t=times[i]-times[0], 
            gamma=gamma, npts=len(xref), dustFrac=0.)
        lbs = ['rho', 'u', 'p', 'ei']

        #fig, axs = plt.subplots(4, 1, figsize=(2, 3), sharex=True)
        #for j, y in enumerate([rho, vx, p, ei]):
        fig, axs = plt.subplots(3, 1, figsize=(1.5, 2), sharex=True)
        for j, y in enumerate([rho, vx, p]):
            axs[j].plot(xref, y, 'tab:blue', lw=3)
            if times[i] >= -0.1:
                if lbs[j] != 'ei':
                    axs[j].plot(xref, values[lbs[j]], 'k--')
                else:
                    axs[j].plot(xref, values['p'] / (gamma - 1) / values['rho'], 'k--')

            axs[j].set_xlim([xref[0], xref[-1]])
            axs[j].set_ylim(ylims[j])
            #axs[i].set_ylabel(' ')
            axs[j].spines['right'].set_visible(False)
            axs[j].spines['top'].set_visible(False)
        #axs[-1].set_xlabel(' ')

        plt.tight_layout()
        plt.savefig(f'pred_{times[i]}.svg')
        #plt.show()

