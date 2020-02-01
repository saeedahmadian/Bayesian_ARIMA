import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.stats import norm

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
tfd = tfp.distributions

sns.set()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
np.random.seed(111)
tf.compat.v1.set_random_seed(111)

###########################################################
"""read data from data folder"""
data = pd.read_csv('data/final_data.csv',header=0,index_col='Date',parse_dates=True)
filter1 = data.DA_LMP > 0
filter2 = data.RT_LMP > 0
filter3 = data.DA_DEMD > 0
filter4 = data.DEMAND > 0
data.where(filter1 & filter2 & filter3 & filter4, inplace=True)
data.interpolate('linear', limit_direction='both')

new_data=data['2015-12-1':'2015-12-20']
# new_data['RT_LMP_1']=new_data.RT_LMP.shift(1)
# new_data['DEMAND_1']=new_data.DEMAND.shift(1)
# new_data.drop(new_data.index[0])
from sklearn.preprocessing import MinMaxScaler
df=MinMaxScaler(feature_range=(0,10)).fit_transform(new_data[["DA_DEMD", "DA_LMP", "RT_LMP"]].dropna())
x= df[:,0:2]
y=df[:,2]

################################################################
class Determinitic_Model(object):
  def __init__(self,N_var=2,N_out=1):
    # Initialize the weights to `5.0` and the bias to `0.0`
    # In practice, these should be initialized to random values (for example, with `tf.random.normal`)
    self.W = tf.Variable(tf.zeros([N_var,N_out]),dtype=tf.float32)
    self.b = tf.Variable(tf.zeros([N_out]),dtype=tf.float32)
  def __call__(self, x):
    return tf.add(tf.matmul(x,self.W),self.b)


def loss_deter(predicted_y, target_y):
  return tf.reduce_mean(tf.square(predicted_y - target_y))

def train(model, inputs, outputs, learning_rate):
    optim = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    with tf.GradientTape() as t:
        current_loss = loss_deter(model(inputs), outputs)
    dW, db = t.gradient(current_loss, [model.W, model.b])
    # grad = tape.gradient(loss, [w, b])
    processed_grad = [gr * 1 for gr in [dW,db]]
    grad_var = zip([dW, db], [model.W, model.b])
    optim.apply_gradients(grad_var)
    # model.W.assign_sub(learning_rate * dW)
    # model.b.assign_sub(learning_rate * db)
    return dW, db


def run_deter():
    model = Determinitic_Model()
    batch_size = 5
    N_step = int(x.shape[0] / batch_size)
    list_W1 = []
    list_W2 = []
    list_b = []
    list_dw1 = []
    list_dw2 = []
    list_db = []
    list_loss = []
    for i in range(N_step):
        x_batch = tf.convert_to_tensor(x[i * batch_size:(i + 1) * batch_size, :], dtype=tf.float32)
        y_batch = tf.convert_to_tensor(y[i * batch_size:(i + 1) * batch_size], dtype=tf.float32)
        list_W1.append(model.W.numpy()[0])
        list_W2.append(model.W.numpy()[1])
        list_b.append(model.b.numpy())
        current_loss = loss_deter(model(x_batch), y_batch)
        list_loss.append(current_loss)
        dloss_dw, dloss_db = train(model, x_batch, y_batch, learning_rate=0.0005)
        list_dw1.append(dloss_dw.numpy()[0])
        list_dw2.append(dloss_dw.numpy()[1])
        list_db.append(dloss_db.numpy())





def determinstic_model(x_data,y_data,batch_size=10):
    lr=.01
    D = x_data.shape[1]
    w= tf.Variable(tf.zeros([D,1]),name='w_deter1',dtype=tf.float32)
    b = tf.Variable(tf.zeros(1),name='b_deter1',dtype=tf.float32)
    # y= tf.matmul(x,w)+b
    N_step= int(x_data.shape[0]/batch_size)
    for i in range(N_step):
        x_batch= tf.convert_to_tensor(x_data[i*batch_size:(i+1)*batch_size,:],dtype=tf.float32)
        y_batch = tf.convert_to_tensor(y_data[i * batch_size:(i + 1) * batch_size], dtype=tf.float32)
        y_pred = tf.add(tf.matmul(x_batch, w), b)
        # y_pred=tf.reshape(y_pred,[batch_size])
        # y_batch = tf.reshape(y_batch,[batch_size])
        with tf.GradientTape(persistent=True) as tape:
            # tape.watch([w,b])
            y_pred = tf.add(tf.matmul(x_batch, w), b)
            loss= tf.keras.losses.MSE(tf.reshape(y_batch,[batch_size]),tf.reshape(y_pred,[batch_size]))
            # loss = tf.reduce_sum(tf.reshape(tf.square(y_batch,y_pred),shape=[batch_size]))
        dw,db = tape.gradient(y_pred,[w,b])
        w.assign_sub(lr*dw)
        b.assign_sub(lr*db)
    return w,b,loss


def determinstic_model2(x_data,y_data,batch_size=10):
    D = x_data.shape[1]
    w = tf.Variable(tf.zeros([D, 1]), 'w_deter',dtype=tf.float32)
    b = tf.Variable(tf.zeros([1]), 'w_deter',dtype=tf.float32)
    N_step= int(x_data.shape[0]/batch_size)
    opt= tf.keras.optimizers.SGD(learning_rate=0.1)
    for i in range(N_step):
        x_batch= tf.convert_to_tensor(x_data[i*batch_size:(i+1)*batch_size,:],dtype=tf.float32)
        y_batch = tf.convert_to_tensor(y_data[i * batch_size:(i + 1) * batch_size,:], dtype=tf.float32)
        with tf.GradientTape() as tape:
            y_pred = tf.matmul(x_batch, w) + b
            # y_pred= determinstic_model(x_batch)
            loss = tf.reduce_sum(tf.square(y_batch,y_pred))
        grad= tape.gradient(loss,[w,b])
        processed_grad=[gr*1 for gr in grad]
        grad_var = zip(processed_grad,[w,b])
        opt.appy_gradients(grad_var)

# output1 = determinstic_model(x,y.reshape(-1,1),10)
# output2 = determinstic_model2(x,np.array(y).reshape(-1,1),10)



import tensorflow.compat.v1 as tf
mygraph=tf.Graph()


def linear_regression(features):
    D = features.shape[1]  # number of dimensions
    coeffs = ed.Normal(  # normal prior on weights
        loc=tf.zeros([D, 1], dtype=tf.float32),
        scale=10*tf.ones([D, 1], dtype=tf.float32),
        name="coeffs")
    bias = ed.Normal(  # normal prior on bias
        loc=tf.zeros([1], dtype=tf.float32),
        scale=10*tf.ones([1], dtype=tf.float32),
        name="bias")
    noise_std = ed.HalfNormal(  # half-normal prior on noise std
        scale=10*tf.ones([1], dtype=tf.float32),
        name="noise_std")
    predictions = ed.Normal(  # normally-distributed noise
        loc=tf.matmul(features, coeffs) + bias,
        scale=noise_std,
        name="predictions")
    return predictions

def linear_regression2(features):
    D = features.shape[1]  # number of dimensions
    factor_coeff_loc= ed.Normal(loc=tf.zeros([D, 1], dtype=tf.float32),
                                  scale=10*tf.ones([D, 1], dtype=tf.float32),
                                  name='factor_coeff_loc')
    factor_coeff_scale= ed.HalfNormal(
                                   scale=10 * tf.ones([D, 1], dtype=tf.float32),
                                   name='factor_coeff_scale')

    coeffs = ed.Normal(  # normal prior on weights
        loc=factor_coeff_loc,
        scale=factor_coeff_scale,
        name="coeffs")
    bias = ed.Normal(  # normal prior on bias
        loc=tf.zeros([1], dtype=tf.float32),
        scale=tf.ones([1], dtype=tf.float32),
        name="bias")
    noise_std = ed.HalfNormal(  # half-normal prior on noise std
        scale=tf.ones([1], dtype=tf.float32),
        name="noise_std")
    predictions = ed.Normal(  # normally-distributed noise
        loc=tf.matmul(features, coeffs) + bias,
        scale=noise_std,
        name="predictions")
    return predictions

# Joint posterior distribution
log_joint = ed.make_log_joint_fn(linear_regression)


# Function to compute the log posterior probability
# def target_log_prob_fn(factor_coeff_loc,factor_coeff_scale,coeffs, bias, noise_std):
#     return log_joint(
#         features=tf.convert_to_tensor(x, dtype=tf.float32),
#         factor_coeff_loc=factor_coeff_loc,
#         factor_coeff_scale=factor_coeff_scale,
#         coeffs=coeffs,
#         bias=bias,
#         noise_std=noise_std,
#         predictions=tf.reshape(tf.convert_to_tensor(y, dtype=tf.float32), [y.shape[0], 1]))

def target_log_prob_fn(coeffs, bias, noise_std):
    return log_joint(
        features=tf.convert_to_tensor(x, dtype=tf.float32),
        coeffs=coeffs,
        bias=bias,
        noise_std=noise_std,
        predictions=tf.reshape(tf.convert_to_tensor(y, dtype=tf.float32), [y.shape[0], 1]))
class Timer:
    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, *args):
        print('Elapsed time: %0.2fs' % (time.time() - self.t0))


# HMC Settings
num_results = int(500)  # number of hmc iterations
n_burnin = int(250)  # number of burn-in steps
step_size = 0.01
num_leapfrog_steps = 10

D = x.shape[1]
# Parameter sizes
coeffs_size = [D, 1]
bias_size = [1]
noise_std_size = [1]
# HMC transition kernel
kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=target_log_prob_fn,
    step_size=step_size,
    num_leapfrog_steps=num_leapfrog_steps)

# Define the chain states
states, kernel_results = tfp.mcmc.sample_chain(
    num_results=num_results,
    num_burnin_steps=n_burnin,
    kernel=kernel,
    current_state=[

        tf.zeros(coeffs_size, name='init_coeffs', dtype=tf.float32),
        tf.zeros(bias_size, name='init_bias', dtype=tf.float32),
        tf.ones(noise_std_size, name='init_noise_std', dtype=tf.float32),
    ])
coeffs, bias, noise_std = states
# with Timer(), tf.Session(mygraph) as sess:
#     [
#         coeffs_,
#         bias_,
#         noise_std_,
#         is_accepted_,
#     ] = sess.run([
#         coeffs,
#         bias,
#         noise_std,
#         kernel_results.is_accepted,
#     ])

# Samples after burn-in
coeffs_samples = coeffs[n_burnin:, :, 0]
bias_samples = bias[n_burnin:]
noise_std_samples = noise_std[n_burnin:]
accepted_samples = kernel_results.is_accepted[n_burnin:]


def chain_plot(data, title='', ax=None):
    '''Plot both chain and posterior distribution'''
    if ax is None:
        ax = plt.gca()
    ax.plot(data,'g')
    ax.title.set_text(title + " HMCMC chain")


def post_plot(data, title='', ax=None, true=None, prc=95):
    '''Plot the posterior distribution given MCMC samples'''
    if ax is None:
        ax = plt.gca()
    sns.kdeplot(data,shade=True, color="r", ax=ax)
    tprc = (100 - prc) / 2
    ax.axvline(x=np.percentile(data, tprc), c= 'm',linestyle='-.')
    ax.axvline(x=np.percentile(data, 100 - tprc), c= 'm',linestyle='-.')
    ax.title.set_text(title + " frequency density")
    if true is not None:
        ax.axvline(x=true)


def chain_post_plot(data, title='', ax=None, true=None):
    '''Plot a chain of MCMC samples'''
    chain_plot(data, title=title, ax=ax[0])
    post_plot(data, title=title, ax=ax[1], true=true)

w_true=[.2,.2]
b_true=.05
noise_std_true=1
# Plot chains and distributions for coefficients
fig, axes = plt.subplots(D + 2, 2, sharex='col', sharey='col')
fig.set_size_inches(6.4, 8)
w_names=['demand coeff','DA price coeff']
for i in range(D):
    chain_post_plot(coeffs_samples[:, i].numpy(),
                    title=w_names[i],
                    ax=axes[i], true=w_true[i])

# Plot chains and distributions for bias
chain_post_plot(bias_samples[:, 0].numpy(),
                title="ARIMA Bias",
                ax=axes[D], true=b_true)

# Plot chains and distributions for noise std dev
chain_post_plot(noise_std_samples[:, 0].numpy(),
                title="ARIMA variance std",
                ax=axes[D + 1], true=noise_std_true)

axes[D + 1][1].set_xlabel("Parameter value")
fig.tight_layout()
plt.show()

a=1





