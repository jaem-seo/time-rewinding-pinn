import deepxde as dde
import numpy as np
import matplotlib
#matplotlib.rcParams['axes.linewidth']=1.5
matplotlib.rcParams['axes.labelsize']=15
matplotlib.rcParams['axes.titlesize']=15
matplotlib.rcParams['xtick.labelsize']=15
matplotlib.rcParams['ytick.labelsize']=15
import matplotlib.pyplot as plt
import pandas as pd

Lx, Ly = 4.0, 4.0
x0, y0 = 2.75, 2.75
xmin, xmax = 0., Lx
ymin, ymax = 0., Ly
tmin, tmax, tobs = 0.05, 1.0, 1.0
n_output = 4 # rho, vx, vy, energy
num_x = 100
gamma = 1.4
obs_path = '2d_solution.csv'

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

geom = dde.geometry.Rectangle([0., 0.], [Lx, Ly])
timedomain = dde.geometry.TimeDomain(tmin, tmax)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

net = dde.nn.FNN([3] + [64] * 4 + [n_output], ["tanh", "tanh", "tanh", "tanh", None], "Glorot normal")
data = dde.data.TimePDE(
    geomtime, 
    pde, 
    [],
    num_domain=0, 
    num_boundary=0, 
)
model = dde.Model(data, net)
model.compile("adam", lr=0.001)
model.restore("saved_model-3404.ckpt")

xtmp, ytmp = np.linspace(xmin, xmax, num_x), np.linspace(ymin, ymax, num_x)
Xtmp, Ytmp = np.meshgrid(xtmp, ytmp)

df = pd.read_csv(obs_path)
r, rho, vr, p = df['r'].values, df['rho'].values, df['v'].values, df['p'].values
theta = np.linspace(0, 2 * np.pi, 101)
R, Theta = np.meshgrid(r, theta)
Rho, _ = np.meshgrid(rho, theta)
P, _ = np.meshgrid(p, theta)
Vr, _ = np.meshgrid(vr, theta)
Vx, Vy = Vr * np.cos(Theta), Vr * np.sin(Theta)
R_fluc = R - 0.15 * np.cos(5 * Theta)
xref, yref = x0 + 1 * np.cos(theta), y0 + 1 * np.sin(theta)
X, Y = x0 + R_fluc * np.cos(Theta), y0 + R_fluc * np.sin(Theta)

# Plot results
fig, axs = plt.subplots(1, 1, figsize=(5, 3))
mappable = axs.contourf(X, Y, P, levels=50, linewidths=0, antialiased=False)
plt.colorbar(mappable=mappable, ticks=[0.0, 0.2, 0.4, 0.6, 0.8])
axs.plot(xref, yref, 'w--', lw=2)
#axs.scatter(x0, y0, c='r', marker='x')
axs.set_title('t=0.00 s')
axs.set_aspect('equal', 'box')
axs.set_xlim([xmin, xmax])
axs.set_ylim([ymin, ymax])
#plt.savefig('blast_obs.svg')
plt.savefig('blast_obs.png')
plt.show()


times = [0.7, 0.5, 0.3]
xs = [np.zeros([num_x ** 2, 3]) for i in range(len(times))]
for i in range(len(times)):
    xs[i][:, 0] = Xtmp.flatten()
    xs[i][:, 1] = Ytmp.flatten()
    xs[i][:, 2] = times[i]

ys = [model.predict(x) for x in xs]


for i, t in enumerate(times):
    rref = t
    xref, yref = x0 + rref * np.cos(theta), y0 + rref * np.sin(theta)
    fig, axs = plt.subplots(1, 1, figsize=(3, 3))
    mappable = axs.contourf(Xtmp, Ytmp, ys[i][:, 3].reshape([num_x, num_x]), levels=50, linewidths=0, antialiased=False)
    #plt.colorbar(mappable=mappable, ticks=[0.0, 0.2, 0.4, 0.6, 0.8])
    axs.plot(xref, yref, 'w--', lw=1.5)
    #if t == 0.1:
    #    axs.scatter(x0, y0, c='r', marker='x')
    axs.set_title(f't={t - tmax:.2f} s')
    axs.set_aspect('equal', 'box')
    #plt.savefig(f'blast_{t - tmax:.2f}.svg')
    plt.savefig(f'blast_{t - tmax:.2f}.png')
    plt.show()
