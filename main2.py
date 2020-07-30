import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from plot_utils import autoscale, get_lims

cfg_init = {
    'value_init': 336000,
    'deposit_monthly': 10000,
    'return_annual_mean': (1+0.008027)**12-1, #0.04,
    'return_monthly_std': 0.054027,
    'standard_rate_annual_mean': 0.005,
    'n_months': 480,
    'annual_raise': 0.03,
    'months_of_interest': [60, 120, 240, 360],
    'n_trajectories': 500
}

with open('kde.pickle', 'rb') as file:
        monthly_return_kde = pickle.load(file)

class Simulator:

    def __init__(self, cfg):
        self.cfg = cfg
        self.allocate()

    def set_cfg(self, cfg):
        self.cfg = cfg
        self.allocate()

    def update_cfg_item(self, key, val):
        if key not in self.cfg.keys():
            error_msg = f"{key} not in cfg"
            raise KeyError(error_msg)
        self.cfg[key] = val
        self.allocate()

    def allocate(self):
        self.value_trajectories = np.zeros((self.cfg['n_trajectories'], self.cfg['n_months']+1))
        self.deposit_trajectories = np.zeros((self.cfg['n_trajectories'], self.cfg['n_months']))
        self.return_trajectories = np.zeros((self.cfg['n_trajectories'], self.cfg['n_months']))
        self.standard_rate_trajectories = np.zeros((self.cfg['n_trajectories'], (self.cfg['n_months']+1)//12))
        self.value_median_trajectory = np.zeros((self.cfg['n_months']+1))
        self.value_mean_trajectory = np.zeros((self.cfg['n_months']+1))
        self.value_95_quant_trajectory = np.zeros((self.cfg['n_months']+1))
        self.value_05_quant_trajectory = np.zeros((self.cfg['n_months']+1))

    def simulate_trajectories(self):

        # Pre-calculations
        # return_monthly_mean = (1+self.cfg['return_annual_mean'])**(1/12)-1
        # return_monthly_std = self.cfg['return_monthly_std']

        self.value_trajectories[:, 0] = self.cfg['value_init']
        self.return_trajectories = self.__sample_monthly_returns() #shape=[n_trajectories, n_months]
        self.deposit_trajectories = self.__sample_monthly_deposits() #shape=[n_trajectories, n_months]
        self.standard_rate_trajectories = self.__sample_annual_standard_rates() #shape=[n_trajectories, (n_months+1)/12]
        for iMonth in range(self.cfg['n_months']):
            self.value_trajectories[:, iMonth+1] = self.value_trajectories[:, iMonth] * (1 + self.return_trajectories[:, iMonth]) + self.deposit_trajectories[:, iMonth]
            if (iMonth%12==0) and (iMonth > 0):
                standard_rates_month = self.standard_rate_trajectories[:, iMonth//12]
                taxes = self.__tax_isk(standard_rates_month, iMonth)
                self.value_trajectories[:, iMonth+1] = self.value_trajectories[:, iMonth+1] - taxes

        return self.value_trajectories, self.return_trajectories, self.deposit_trajectories

    def calculate_statistical_trajectories(self):
        self.value_mean_trajectory = np.mean(self.value_trajectories, axis=0)
        self.value_median_trajectory, self.value_05_quant_trajectory, self.value_95_quant_trajectory = np.quantile(self.value_trajectories, [0.5, 0.05, 0.95], axis=0)
        return self.value_mean_trajectory, self.value_median_trajectory, self.value_05_quant_trajectory, self.value_95_quant_trajectory

    def get_shit(self):
        #self.allocate()
        self.simulate_trajectories()
        self.calculate_statistical_trajectories()    
        return self.value_mean_trajectory, self.value_05_quant_trajectory, self.value_95_quant_trajectory, self.deposit_trajectories[0,:], self.cfg['value_init']

    def __sample_monthly_returns(self):
        n = self.cfg['n_trajectories']*self.cfg['n_months'] 
        return monthly_return_kde.resample(n).reshape(self.cfg['n_trajectories'], self.cfg['n_months'])

    def __sample_monthly_deposits(self):
        monthly_deposit_trajectory = self.cfg['deposit_monthly']*(1+self.cfg['annual_raise'])**(np.arange(self.cfg['n_months']) // 12)
        return np.repeat(monthly_deposit_trajectory[np.newaxis, :], self.cfg['n_trajectories'], axis=0)

    def __sample_annual_standard_rates(self):
        return self.cfg['standard_rate_annual_mean']*np.ones((self.cfg['n_trajectories'], (self.cfg['n_months']+1)//12))

    def __tax_isk(self, standard_rates_month, month_idx):
        """ 
            Parameters
            --------------
            standard_rate : np.ndarray, shape=[n_trajectories]
        """
        tax_rate = np.maximum(0.0125, 0.01+standard_rates_month)
        return tax_rate*( \
            self.value_trajectories[:, month_idx-12]+ \
            self.value_trajectories[:, month_idx-9]+ \
            self.value_trajectories[:, month_idx-6]+ \
            self.value_trajectories[:, month_idx-3]+ \
            self.deposit_trajectories[:, month_idx-12:month_idx].sum(axis=1) \
            ) / 4



class MainWindow:

    MAX_NO_SLIDERS = 20

    def __init__(self, cfg):
        self.simulator = Simulator(cfg)
        self.__initialize_figure(cfg['n_months'])
        val, val_low, val_high, dep, val_init = self.simulator.get_shit()
        self.__plot_trajectory(val, val_low, val_high, dep, val_init)
        plt.show()

    def __add_slider(self, key, vallims=None, valinit=None, valsteporder=None, title=None, **kwargs):

        def get_slider_lims(val):
            high = val*2
            low = val/2
            return low, high

        if valinit is None:
            valinit = self.simulator.cfg[key]

        if vallims is None:
            vallims = get_slider_lims(valinit)

        if valsteporder is None:
            valsteporder = -2
        valstep = 10**valsteporder

        if title is None: 
            title = key

        valfmt = "%d" if valsteporder > 0 else f"%1.{-valsteporder}f"

        if self.slider_counter >= self.MAX_NO_SLIDERS:
            raise Error("Too many sliders")
        ax_slider = self.fig.add_subplot(self.MAX_NO_SLIDERS, 2, self.slider_counter*2+1)
        self.slider_counter += 1
        self.sliders[key] = Slider(ax_slider, title, vallims[0], vallims[1], valinit=valinit, valstep=valstep, orientation='horizontal', valfmt=valfmt, **kwargs)
        on_changed = lambda x: self.__recalculate_and_draw(key, x)
        self.sliders[key].on_changed(on_changed)

    def __initialize_figure(self, n_months):

        self.fig = plt.figure(figsize=(16,9))

        self.ax_plot = self.fig.add_subplot(1,2,2)
        self.ax_plot.grid(axis='x', which='major')
        self.ax_plot.grid(axis='y', which='both')
        self.ax_plot.set_xlabel("Years from now")
        self.ax_plot.set_ylabel("Capital (kr)")
        self.ax_plot.set_xlim([0, (n_months+1)/12])
        
        self.trajectory_line = None
        self.trajectory_line_low = None
        self.trajectory_line_high = None
        self.deposit_line = None

        self.sliders = {}
        self.slider_counter = 0

        self.__add_slider('deposit_monthly', title="Monthly deposit (kr)", vallims=[0,20000], valsteporder=2)
        self.__add_slider('return_annual_mean', title="Annual return", vallims=[-0.10, 0.20], valsteporder=-3)
        self.__add_slider('annual_raise', title="Annual raise", vallims=[0, 0.05], valsteporder=-3)
        self.__add_slider('value_init', title="Initial value (kr)", vallims=[0, 500000], valsteporder=4)
        self.__add_slider('standard_rate_annual_mean', title="Standard rate", vallims=[-0.03, 0.06], valsteporder=-3)

    def __recalculate_and_draw(self, key, val):
        self.simulator.update_cfg_item(key, val)
        val_mean, val_05, val_95, dep, val_init = self.simulator.get_shit()
        self.__plot_trajectory(val_mean, val_05, val_95, dep, val_init)

    def __plot_trajectory(self, val, val_low, val_high, dep, value_init):
        if self.trajectory_line is not None:
            self.trajectory_line.set_ydata(val)
        else:
            self.trajectory_line = self.ax_plot.semilogy([i/12 for i in range(len(val))], val)[0]

        if self.trajectory_line_low is not None:
            self.trajectory_line_low.set_ydata(val_low)
        else:
            self.trajectory_line_low = self.ax_plot.semilogy([i/12 for i in range(len(val_low))], val_low)[0]

        if self.trajectory_line_high is not None:
            self.trajectory_line_high.set_ydata(val_high)
        else:
            self.trajectory_line_high = self.ax_plot.semilogy([i/12 for i in range(len(val_high))], val_high)[0]

        if self.deposit_line is not None:
            self.deposit_line.set_ydata(value_init+np.cumsum(np.array(dep)))
        else:
            self.deposit_line = self.ax_plot.semilogy([i/12 for i in range(len(dep))], value_init+np.cumsum(np.array(dep)))[0]

        # Update scale if necessary
        lims_y = self.ax_plot.get_ylim()
        if (max(val_high) > lims_y[1]*0.9) or (max(val_high) < lims_y[1] / 10) or (min(val_low) < lims_y[0]):
            autoscale(self.ax_plot, axis='y', factor=2)


def main(cfg):
    main_window = MainWindow(cfg)

if __name__=="__main__":
    main(cfg_init)
