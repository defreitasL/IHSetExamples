from IHSetYates09 import cal_Yates09, yates09
from IHSetCalibration import setup_spotpy, mielke_skill_score
import xarray as xr
import os
import matplotlib.pyplot as plt
import spotpy as spt
import numpy as np
from IHSetExamples import plot_par_evolution

# Avaliable methods: NSGAII, mle, mc, dds, mcmc, sa, abc, lhs, rope, sceua, demcz, padds, fscabc

# config = xr.Dataset(coords={'dt': 3,                # [hours]
#                             'switch_Yini': 0,       # Calibrate the initial position? (0: No, 1: Yes)
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

config = xr.Dataset(coords={'dt': 3,                # [hours]
                            'switch_Yini': 0,       # Calibrate the initial position? (0: No, 1: Yes)
                            'Ysi': 1999,            # Initial year for calibration
                            'Msi': 1,               # Initial month for calibration
                            'Dsi': 1,               # Initial day for calibration
                            'Ysf': 2010,            # Final year for calibration
                            'Msf': 1,               # Final month for calibration
                            'Dsf': 1,               # Final day for calibration
                            'cal_alg': 'sceua',     # Avaliable methods: sceua
                            'metrics': 'mss',       # Metrics to be minimized (mss, RP, rmse, nsse)
                            'repetitions': 40000    # Number of repetitions for the calibration algorithm
                            })

wrkDir = os.getcwd()
config.to_netcdf(wrkDir+'/data/config.nc', engine='netcdf4')

# Verifica se o arquivo existe
model = cal_Yates09(wrkDir+'/data/')

setup = setup_spotpy(model)

results = setup.setup()

bestindex, bestobjf = spt.analyser.get_minlikeindex(results)
best_model_run = results[bestindex]
fields=[word for word in best_model_run.dtype.names if word.startswith('sim')]
best_simulation = list(best_model_run[fields])

a = -best_model_run['para']
b = best_model_run['parb']
cacr = -best_model_run['parcacr']
cero = -best_model_run['parcero']

full_run, _ = yates09(model.E, model.dt, a, b, cacr, cero, model.Obs[0])

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
ax.plot(model.time, full_run, color='red',linestyle='solid', label= 'Yates et al.(2009)')
plt.fill([model.start_date, model.end_date, model.end_date, model.start_date], [-1e+5, -1e+5, 1e+5, 1e+5], 'k', alpha=0.1, edgecolor=None, label = 'Calibration Period')
plt.ylim([40,80])
plt.xlim([model.time[0], model.time[-1]])
plt.ylabel('Shoreline position [m]', fontdict=font)
plt.legend(ncol = 6,prop={'size': 6}, loc = 'upper center', bbox_to_anchor=(0.5, 1.15))
plt.grid(visible=True, which='both', linestyle = '--', linewidth = 0.5)
fig.savefig('./results/Yates09_Best_modelrun_'+str(config.cal_alg.values)+'.png',dpi=300)

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