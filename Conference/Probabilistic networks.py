import logging
import os

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim as optim

pyro.set_rng_seed(1)
assert pyro.__version__.startswith('1.2.1')
sns.set()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# %config InlineBackend.figure_format = 'svg'
np.random.seed(111)
# tf.set_random_seed(111)


###########################################################
"""read data from data folder"""
data = pd.read_csv('data/final_data.csv',header=0,index_col='Date',parse_dates=True)
filter1 = data.DA_LMP > 0
filter2 = data.RT_LMP > 0
filter3 = data.DA_DEMD > 0
filter4 = data.DEMAND > 0
data.where(filter1 & filter2 & filter3 & filter4, inplace=True)
data.interpolate('linear', limit_direction='both')

new_data=data['2015-12-1':'2015-12-7']
new_data['RT_LMP_1']=new_data.RT_LMP.shift(1)
new_data['DEMAND_1']=new_data.DEMAND.shift(1)
new_data.drop(new_data.index[0])

################################################################


################################################################

def model(DA_DEMD, DA_LMP, RT_LMP):
    bias = pyro.sample("bias", dist.Normal(0., 10.))
    w_demand = pyro.sample("w_demand", dist.Normal(0., 1.))
    w_lmp = pyro.sample("w_lmp", dist.Normal(0., 1.))
    sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
    mean = bias + w_demand * DA_DEMD + w_lmp * DA_LMP
    with pyro.plate("data", len(RT_LMP)):
        pyro.sample("obs", dist.Normal(mean, sigma), obs=RT_LMP)

def guide(DA_DEMD, DA_LMP, RT_LMP):
    bias_loc = pyro.param('bias_loc', torch.tensor(0.))
    bias_scale = pyro.param('bias_scale', torch.tensor(1.),
                         constraint=constraints.positive)
    sigma_loc = pyro.param('sigma_loc', torch.tensor(1.),
                             constraint=constraints.positive)
    weights_loc = pyro.param('weights_loc', torch.randn(2))
    weights_scale = pyro.param('weights_scale', torch.ones(2),
                               constraint=constraints.positive)
    bias = pyro.sample("bias", dist.Normal(bias_loc, bias_scale))
    w_demand = pyro.sample("w_demand", dist.Normal(weights_loc[0], weights_scale[0]))
    w_lmp = pyro.sample("w_lmp", dist.Normal(weights_loc[1], weights_scale[1]))
    # b_ar = pyro.sample("bAR", dist.Normal(weights_loc[2], weights_scale[2]))
    sigma = pyro.sample("sigma", dist.Normal(sigma_loc, torch.tensor(0.05)))
    mean = bias + w_demand * DA_DEMD + w_lmp * DA_LMP

def summary(samples):
    site_stats = {}
    for site_name, values in samples.items():
        marginal_site = pd.DataFrame(values)
        describe = marginal_site.describe(percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()
        site_stats[site_name] = describe[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    return site_stats

# Prepare training data
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
df=MinMaxScaler(feature_range=(0,10)).fit_transform(new_data[["DA_DEMD", "DA_LMP", "RT_LMP"]].dropna())
# df = new_data[["DA_DEMD", "DA_LMP", "RT_LMP"]]
# df = df[np.isfinite(df.rgdppc_2000)]
# df["rgdppc_2000"] = np.log(df["rgdppc_2000"])
train = torch.tensor(df, dtype=torch.float)


###############################################################################
###############################################################################
from pyro.infer import SVI, Trace_ELBO


svi = SVI(model,
          guide,
          optim.Adam({"lr": .05}),
          loss=Trace_ELBO())

DA_DEMD, DA_LMP, RT_LMP = train[:, 0], train[:, 1], train[:, 2]
pyro.clear_param_store()
num_iters = 5000
list_loss=[]
for i in range(num_iters):
    elbo = svi.step(DA_DEMD, DA_LMP, RT_LMP)
    list_loss.append(elbo)
    if i % 500 == 0:
        logging.info("Elbo loss: {}".format(elbo))

##############################################################################
###############################################################################





################################################################################
################################################################################
from pyro.infer import Predictive


num_samples = 1000
predictive = Predictive(model, guide=guide, num_samples=num_samples)
svi_samples = {k: v.reshape(num_samples).detach().cpu().numpy()
               for k, v in predictive(DA_DEMD, DA_LMP, RT_LMP).items()
               if k != "obs"}

#####################################################################################

for site, values in summary(svi_samples).items():
    print("Site: {}".format(site))
    print(values, "\n")

######################################################################################
from pyro.infer import MCMC, NUTS


nuts_kernel = NUTS(model)

mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
mcmc.run(is_cont_africa, ruggedness, log_gdp)

hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

###########################################################################################

for site, values in summary(hmc_samples).items():
    print("Site: {}".format(site))
    print(values, "\n")

############################################################################################
sites = ["a", "bA", "bR", "bAR", "sigma"]

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.suptitle("Marginal Posterior density - Regression Coefficients", fontsize=16)
for i, ax in enumerate(axs.reshape(-1)):
    site = sites[i]
    sns.distplot(svi_samples[site], ax=ax, label="SVI (DiagNormal)")
    sns.distplot(hmc_samples[site], ax=ax, label="HMC")
    ax.set_title(site)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right');
#################################################################################################

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
fig.suptitle("Cross-section of the Posterior Distribution", fontsize=16)
sns.kdeplot(hmc_samples["bA"], hmc_samples["bR"], ax=axs[0], shade=True, label="HMC")
sns.kdeplot(svi_samples["bA"], svi_samples["bR"], ax=axs[0], label="SVI (DiagNormal)")
axs[0].set(xlabel="bA", ylabel="bR", xlim=(-2.5, -1.2), ylim=(-0.5, 0.1))
sns.kdeplot(hmc_samples["bR"], hmc_samples["bAR"], ax=axs[1], shade=True, label="HMC")
sns.kdeplot(svi_samples["bR"], svi_samples["bAR"], ax=axs[1], label="SVI (DiagNormal)")
axs[1].set(xlabel="bR", ylabel="bAR", xlim=(-0.45, 0.05), ylim=(-0.15, 0.8))
handles, labels = axs[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right');

########################################################################################################


a=1