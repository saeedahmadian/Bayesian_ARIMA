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
# new_data['RT_LMP_1']=new_data.RT_LMP.shift(1)
# new_data['DEMAND_1']=new_data.DEMAND.shift(1)
# new_data.drop(new_data.index[0])

################################################################


################################################################

def model(DA_DEMD, DA_LMP, RT_LMP):
    bias = pyro.sample("bias", dist.Normal(0., 10.))
    w_demand = pyro.sample("w_demand", dist.Normal(0., 10))
    w_lmp = pyro.sample("w_lmp", dist.Normal(0., 10))
    sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
    mean = bias + w_demand * DA_DEMD + w_lmp * DA_LMP
    pyro.sample("pred", dist.Normal(mean, sigma))
    with pyro.plate("data", len(RT_LMP)):
        pyro.sample("obs", dist.Normal(mean, sigma), obs=RT_LMP)


def guide(DA_DEMD, DA_LMP, RT_LMP):
    bias_loc = pyro.param('bias_loc', torch.tensor(0.))
    bias_scale = pyro.param('bias_scale', torch.tensor(10),
                         constraint=constraints.positive)
    sigma_loc = pyro.param('sigma_loc', torch.tensor(1.),
                             constraint=constraints.positive)
    sigma_scale = pyro.param('sigma_scale', torch.tensor(1),
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


###################################################################
from sklearn.preprocessing import MinMaxScaler
df=MinMaxScaler(feature_range=(0,10)).fit_transform(new_data[["DA_DEMD", "DA_LMP", "RT_LMP"]].dropna())
# df = new_data[["DA_DEMD", "DA_LMP", "RT_LMP"]]
# df = df[np.isfinite(df.rgdppc_2000)]
# df["rgdppc_2000"] = np.log(df["rgdppc_2000"])
train = torch.tensor(df, dtype=torch.float)


###############################################################################
from pyro.infer.autoguide import AutoDiagonalNormal
guide = AutoDiagonalNormal(model)

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


###############################################################
# guide.requires_grad_(False)

for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name))

################################################################

from pyro.infer import Predictive


def summary(samples):
    site_stats = {}
    for k, v in samples.items():
        v = torch.tensor(v)
        site_stats[k] = {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
            "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
        }
    return site_stats

n_sample=800


predictive = Predictive(model, guide=guide, num_samples=n_sample,
                        return_sites=("bias","w_demand","w_lmp","sigma","obs", "_RETURN"))
samples = predictive(train[:, 0], train[:, 1], train[:, 2])

# predictive = Predictive(model, guide=guide, num_samples=n_sample)
# samples = {k: v.reshape(n_sample).detach().cpu().numpy()
#                for k, v in predictive(train[:, 0], train[:, 1], train[:, 2]).items()
#                if k != "obs"}

pred_summary = summary(samples)


mu = pred_summary["_RETURN"]
y = pred_summary["obs"]
predictions = pd.DataFrame({
    "demand": train[:, 0],
    "lmp": train[:, 1],
    "mu_mean": mu["mean"],
    "mu_perc_5": mu["5%"],
    "mu_perc_95": mu["95%"],
    "y_mean": y["mean"],
    "y_perc_5": y["5%"],
    "y_perc_95": y["95%"],
    "true_gdp": train[:,2],
})

a=1