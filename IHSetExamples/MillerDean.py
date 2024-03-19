from IHSetMillerDean import cal_MillerDean, millerDean
from IHSetCalibration import setup_spotpy, mielke_skill_score
import xarray as xr
import os
import matplotlib.pyplot as plt
import spotpy as spt
import numpy as np
from IHSetExamples import plot_par_evolution
from IHSetUtils import BreakingPropagation

# Avaliable methods: NSGAII, mle, mc, dds, mcmc, sa, abc, lhs, rope, sceua, demcz, padds, fscabc

# config = xr.Dataset(coords={'dt': 3,                # [hours]
#                             'switch_Yini': 0,       # Calibrate the initial position? (0: No, 1: Yes)
#                             'switch_vlt': 0,        # Calibrate the longterm trend? (0: No, 1: Yes)
#                             'vlt': 0,               # Longterm trend [m]
#                             'Ysi': 2000,            # Initial year for calibration
#                             'Msi': 1,               # Initial month for calibration
#                             'Dsi': 1,               # Initial day for calibration
#                             'Ysf': 2005,            # Final year for calibration
#                             'Msf': 1,               # Final month for calibration
#                             'Dsf': 1,               # Final day for calibration
#                             'cal_alg': 'NSGAII',    # Avaliable methods: NSGAII
#                             'metrics': 'mss_nsse',  # Metrics to be minimized (mss_rmse, mss_rho, mss_rmse_rho)
#                             'n_pop': 50,            # Number of individuals in the population
#                             'n_obj': 2,             # Number of objectives to be minimized 
#                             'generations': 1000,    # Number of generations for the calibration algorithm
#                             })              
config = xr.Dataset(coords={'dt': 3,                  # [hours]
                            'depth': 10,              # Water depth [m]
                            'D50': .3e-3,              # Median grain size [m]
                            'bathy_angle': 54.8,      # Bathymetry mean orientation [deg N]
                            'break_type': 'spectral', # Breaking type (spectral or linear)
                            'Hberm': 1,               # Berm height [m]
                            'flagP': 1,               # Parameter Proportionality
                            'switch_Yini': 0,         # Calibrate the initial position? (0: No, 1: Yes)
                            'Ysi': 1999,              # Initial year for calibration
                            'Msi': 1,                 # Initial month for calibration
                            'Dsi': 1,                 # Initial day for calibration
                            'Ysf': 2010,              # Final year for calibration
                            'Msf': 1,                 # Final month for calibration
                            'Dsf': 1,                 # Final day for calibration
                            'cal_alg': 'NSGAII',    # Avaliable methods: NSGAII
                            'metrics': 'mss_nsse',  # Metrics to be minimized (mss_rmse, mss_rho, mss_rmse_rho)
                            'n_pop': 50,            # Number of individuals in the population
                            'n_obj': 2,             # Number of objectives to be minimized 
                            'generations': 1000,    # Number of generations for the calibration algorithm
                            })

# config = xr.Dataset(coords={'dt': 3,                  # [hours]
#                             'depth': 10,              # Water depth [m]
#                             'D50': .3e-3,              # Median grain size [m]
#                             'bathy_angle': 54.8,      # Bathymetry mean orientation [deg N]
#                             'break_type': 'spectral', # Breaking type (spectral or linear)
#                             'Hberm': 1,               # Berm height [m]
#                             'flagP': 1,               # Parameter Proportionality
#                             'switch_Yini': 0,         # Calibrate the initial position? (0: No, 1: Yes)
#                             'Ysi': 1999,              # Initial year for calibration
#                             'Msi': 1,                 # Initial month for calibration
#                             'Dsi': 1,                 # Initial day for calibration
#                             'Ysf': 2010,              # Final year for calibration
#                             'Msf': 1,                 # Final month for calibration
#                             'Dsf': 1,                 # Final day for calibration
#                             'cal_alg': 'sceua',       # Avaliable methods: sceua
#                             'metrics': 'nsse',         # Metrics to be minimized (mss, RP, rmse, nsse)
#                             'repetitions': 50000      # Number of repetitions for the calibration algorithm
#                             })

wrkDir = os.getcwd()
config.to_netcdf(wrkDir+'/data/config.nc', engine='netcdf4')

wav = xr.open_dataset(wrkDir+'/data/wav.nc')

Hb, Dirb, depthb = BreakingPropagation(wav['Hs'].values,
                                       wav['Tp'].values,
                                       wav['Dir'].values,
                                       np.full_like(wav['Hs'].values, config['depth'].values),
                                       np.full_like(wav['Hs'].values, config['bathy_angle'].values),
                                       config['break_type'].values)

wav['Hb'] = xr.DataArray(Hb, dims = 'Y', coords = {'Y': wav['Y']})
wav['Dirb'] = xr.DataArray(Dirb, dims = 'Y', coords = {'Y': wav['Y']})
wav['depthb'] = xr.DataArray(depthb, dims = 'Y', coords = {'Y': wav['Y']})

wav.to_netcdf(wrkDir+'/data/wavb.nc', engine='netcdf4')
wav.close()

model = cal_MillerDean(wrkDir+'/data/')

setup = setup_spotpy(model)

results = setup.setup()

bestindex, bestobjf = spt.analyser.get_minlikeindex(results)
best_model_run = results[bestindex]
fields=[word for word in best_model_run.dtype.names if word.startswith('sim')]
best_simulation = list(best_model_run[fields])

kero = best_model_run['parkero']
kacr = best_model_run['parkacr']
Y0 = best_model_run['parY0']

full_run, yeq = millerDean(model.Hb,
                         model.depthb,
                         model.sl,
                         model.wast,
                         model.dt,
                         model.Hberm,
                         Y0,
                         kero,
                         kacr,
                         model.Obs[0],
                         model.flagP,
                         model.Omega)

plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'font.weight': 'bold'})
font = {'family': 'serif',
        'weight': 'bold',
        'size': 8}
fig = plt.figure(figsize=(12, 2), dpi=300, linewidth=5, edgecolor="#04253a")
ax = plt.subplot(1,1,1)
# ax.plot(best_simulation,color='black',linestyle='solid', label='Best objf.='+str(bestobjf))
ax.scatter(model.time_obs, model.Obs,s = 1, c = 'grey', label = 'Observed data')
ax.plot(model.time, full_run, color='red',linestyle='solid', label= 'Miller and Dean (2004)')
plt.fill([model.start_date, model.end_date, model.end_date, model.start_date], [-1e+5, -1e+5, 1e+5, 1e+5], 'k', alpha=0.1, edgecolor=None, label = 'Calibration Period')
plt.ylim([40,80])
plt.xlim([model.time[0], model.time[-1]])
plt.ylabel('Shoreline position [m]', fontdict=font)
plt.legend(ncol = 6,prop={'size': 6}, loc = 'upper center', bbox_to_anchor=(0.5, 1.15))
plt.grid(visible=True, which='both', linestyle = '--', linewidth = 0.5)
fig.savefig('./results/MillerDean_Best_modelrun_'+str(config.cal_alg.values)+'.png',dpi=300)

# Calibration:
rmse = spt.objectivefunctions.rmse(model.observations, best_simulation)
nsse = spt.objectivefunctions.nashsutcliffe(model.observations, best_simulation)
mss = mielke_skill_score(model.observations, best_simulation)
rp = spt.objectivefunctions.rsquared(model.observations, best_simulation)
bias = spt.objectivefunctions.bias(model.observations, best_simulation)

# Validation:
run_cut = full_run[model.idx_validation]
rmse_v = spt.objectivefunctions.rmse(model.Obs[model.idx_validation_obs], run_cut[model.idx_validation_for_obs])
nsse_v = spt.objectivefunctions.nashsutcliffe(model.Obs[model.idx_validation_obs], run_cut[model.idx_validation_for_obs])
mss_v = mielke_skill_score(model.Obs[model.idx_validation_obs], run_cut[model.idx_validation_for_obs])
rp_v = spt.objectivefunctions.rsquared(model.Obs[model.idx_validation_obs], run_cut[model.idx_validation_for_obs])
bias_v = spt.objectivefunctions.bias(model.Obs[model.idx_validation_obs], run_cut[model.idx_validation_for_obs])

print('Metrics                       | Calibration  | Validation|')
print('RMSE [m]                      | %-5.2f        | %-5.2f     |' % (rmse, rmse_v))
print('Nash-Sutcliffe coefficient [-]| %-5.2f        | %-5.2f     |' % (nsse, nsse_v))
print('Mielke Skill Score [-]        | %-5.2f        | %-5.2f     |' % (mss, mss_v))
print('R2 [-]                        | %-5.2f        | %-5.2f     |' % (rp, rp_v))
print('Bias [m]                      | %-5.2f        | %-5.2f     |' % (bias, bias_v))


plot_par_evolution(results)
config.close()