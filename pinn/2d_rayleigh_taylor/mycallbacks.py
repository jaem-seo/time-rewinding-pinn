import numpy as np
from deepxde.callbacks import Callback
import matplotlib.pyplot as plt

class MyModelCheckpoint(Callback):

    def __init__(
        self,
        filepath,
        verbose=0,
        save_better_only=False,
        period=1,
        monitor="train loss",
        start_step=1
    ):
        super().__init__()
        self.filepath = filepath
        self.verbose = verbose
        self.save_better_only = save_better_only
        self.period = period

        self.monitor = monitor
        self.monitor_op = np.less
        self.epochs_since_last_save = 0
        self.best = np.Inf
        self.steps = 0
        self.start_step = start_step

    def on_epoch_end(self):
        self.steps += 1
        self.epochs_since_last_save += 1
        if self.steps < self.start_step:
            return
        if self.epochs_since_last_save < self.period:
            return
        self.epochs_since_last_save = 0
        if self.save_better_only:
            current = self.get_monitor_value()
            if self.monitor_op(current, self.best):
                try:
                    self.model.saver.save(self.model.sess, self.filepath + ".ckpt")
                    if self.verbose > 0:
                        print(
                            "Epoch {}: {} improved from {:.2e} to {:.2e}, saving model to {} ...\n".format(
                                self.model.train_state.epoch,
                                self.monitor,
                                self.best,
                                current,
                                self.filepath + ".ckpt",
                            )
                        )
                    self.best = current
                except:
                    print("Saving error!!!")
                
        else:
            self.model.save(self.filepath, verbose=self.verbose)

    def get_monitor_value(self):
        if self.monitor == "train loss":
            result = sum(self.model.train_state.loss_train)
        elif self.monitor == "test loss":
            result = sum(self.model.train_state.loss_test)
        else:
            raise ValueError("The specified monitor function is incorrect.")

        return result

class PlotResult(Callback):

    def __init__(self, xmin, xmax, ymin, ymax, tmin, tmax, period=20000):
        super().__init__()
        self.period = period
        self.epochs = 0
        self.steps = 0
        self.xmin, self.xmax, self.ymin, self.ymax, self.tmin, self.tmax = xmin, xmax, ymin, ymax, tmin, tmax

    def on_epoch_end(self):
        self.epochs += 1
        self.steps += 1
        if self.steps < self.period:
            return
        self.steps = 0

        num_x = 101

        xtmp, ytmp = np.linspace(self.xmin, self.xmax, num_x), np.linspace(self.ymin, self.ymax, num_x)
        Xtmp, Ytmp = np.meshgrid(xtmp, ytmp)

        times = np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * (self.tmax - self.tmin) + self.tmin
        xs = [np.zeros([num_x ** 2, 3]) for i in range(len(times))]
        for i in range(len(times)):
            xs[i][:, 0] = Xtmp.flatten()
            xs[i][:, 1] = Ytmp.flatten()
            xs[i][:, 2] = times[i]

        ys = [self.model.predict(x) for x in xs]

        fig, axs = plt.subplots(1, len(times), figsize=(12, 4), sharex=True, sharey=True)

        for i in range(len(times)):
            cs = axs[i].contourf(Xtmp, Ytmp, ys[i][:, 0].reshape([num_x, num_x]))
            axs[i].set_title(f't={times[i]:.3f}')
            fig.colorbar(cs, ax=axs[i])

        plt.tight_layout()
        #plt.show()
        plt.savefig(f'snap_{self.epochs}.png')
