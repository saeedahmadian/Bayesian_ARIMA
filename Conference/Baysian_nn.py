import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.dummy import DummyRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, make_scorer,mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow_probability.python.math import random_rademacher
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('data/final_data.csv',header=0,index_col='Date',parse_dates=True)
filter1 = data.DA_LMP > 0
filter2 = data.RT_LMP > 0
filter3 = data.DA_DEMD > 0
filter4 = data.DEMAND > 0
data.where(filter1 & filter2 & filter3 & filter4, inplace=True)
data.interpolate('linear', limit_direction='both')

new_data=data['2015-12-1':'2015-12-20']
df=MinMaxScaler(feature_range=(0,10)).fit_transform(new_data[["DA_DEMD", "DA_LMP", "RT_LMP"]].dropna())
x= df[:,0:2]
y=df[:,2]
tf.keras.backend.set_floatx('float64')
# Xavier initializer
def xavier(shape):
    return tf.random.truncated_normal(
        shape,
        dtype=tf.float64,
        mean=0.0,
        stddev=np.sqrt(2/sum(shape)))

class BayesianDenseLayer(tf.keras.Model):
    """A fully-connected Bayesian neural network layer

    Parameters
    ----------
    d_in : int
        Dimensionality of the input (# input features)
    d_out : int
        Output dimensionality (# units in the layer)
    name : str
        Name for the layer

    Attributes
    ----------
    losses : tensorflow.Tensor
        Sum of the Kullback–Leibler divergences between
        the posterior distributions and their priors

    Methods
    -------
    call : tensorflow.Tensor
        Perform the forward pass of the data through
        the layer
    """

    def __init__(self, d_in, d_out, name=None):

        super(BayesianDenseLayer, self).__init__(name=name)
        self.d_in = d_in
        self.d_out = d_out

        self.w_loc = tf.Variable(xavier([d_in, d_out]), name='w_loc',dtype=tf.float64)
        self.w_std = tf.Variable(xavier([d_in, d_out]) , name='w_std',dtype=tf.float64)
        self.b_loc = tf.Variable(xavier([1, d_out]), name='b_loc',dtype=tf.float64)
        self.b_std = tf.Variable(xavier([1, d_out]) , name='b_std',dtype=tf.float64)

    def call(self, x, sampling=True):

        """Perform the forward pass"""

        if sampling:

            # Flipout-estimated weight samples
            s = random_rademacher(tf.shape(x),dtype=tf.float64)
            r = random_rademacher([x.shape[0], self.d_out],dtype=tf.float64)
            w_samples = tf.nn.softplus(self.w_std) * tf.random.normal([self.d_in, self.d_out],dtype=tf.float64)
            w_perturbations = r * tf.matmul(x * s, w_samples)
            w_outputs = tf.matmul(x, self.w_loc) + w_perturbations

            # Flipout-estimated bias samples
            r = random_rademacher([x.shape[0], self.d_out],dtype=tf.float64)
            b_samples = tf.nn.softplus(self.b_std) * tf.random.normal([self.d_out],dtype=tf.float64)
            b_outputs = self.b_loc + r * b_samples

            return w_outputs + b_outputs

        else:
            return x @ self.w_loc + self.b_loc

    # def samples_w_b(self,N_sample=100):
    #     w = tfd.Normal(self.w_loc, tf.nn.softplus(self.w_std)).sample()
    #     b = tfd.Normal(self.b_loc, tf.nn.softplus(self.b_std)) .sample()
    #     w_matrix=

    @property
    def losses(self):
        """Sum of the KL divergences between priors + posteriors"""
        weight = tfd.Normal(self.w_loc, tf.nn.softplus(self.w_std))
        bias = tfd.Normal(self.b_loc, tf.nn.softplus(self.b_std))
        prior = tfd.Normal(loc=tf.zeros(1,dtype=tf.float64), scale=1*tf.ones(1,dtype=tf.float64))
        return (tf.reduce_sum(tfd.kl_divergence(weight, prior)) +
                tf.reduce_sum(tfd.kl_divergence(bias, prior)))


class BayesianDenseNetwork(tf.keras.Model):
    """A multilayer fully-connected Bayesian neural network

    Parameters
    ----------
    dims : List[int]
        List of units in each layer
    name : str
        Name for the network

    Attributes
    ----------
    losses : tensorflow.Tensor
        Sum of the Kullback–Leibler divergences between
        the posterior distributions and their priors,
        over all layers in the network

    Methods
    -------
    call : tensorflow.Tensor
        Perform the forward pass of the data through
        the network
    """

    def __init__(self, dims, name=None):

        super(BayesianDenseNetwork, self).__init__(name=name)

        self.steps = []
        self.acts = []
        for i in range(len(dims) - 1):
            self.steps += [BayesianDenseLayer(dims[i], dims[i + 1])]
            self.acts += [tf.nn.relu]

        self.acts[-1] = lambda x: x

    def call(self, x, sampling=True):
        """Perform the forward pass"""

        for i in range(len(self.steps)):
            x = self.steps[i](x, sampling=sampling)
            x = self.acts[i](x)

        return x

    @property
    def losses(self):
        """Sum of the KL divergences between priors + posteriors"""
        return tf.reduce_sum([s.losses for s in self.steps])





class BayesianDensityNetwork(tf.keras.Model):
    """Multilayer fully-connected Bayesian neural network, with
    two heads to predict both the mean and the standard deviation.

    Parameters
    ----------
    units : List[int]
        Number of output dimensions for each layer
        in the core network.
    units : List[int]
        Number of output dimensions for each layer
        in the head networks.
    name : None or str
        Name for the layer
    """

    def __init__(self, units, head_units, name=None):
        # Initialize
        super(BayesianDensityNetwork, self).__init__(name=name)

        # Create sub-networks
        self.core_net = BayesianDenseNetwork(units)
        self.loc_net = BayesianDenseNetwork([units[-1]] + head_units)
        self.std_net = BayesianDenseNetwork([units[-1]] + head_units)

    def call(self, x, sampling=True):
        """Pass data through the model

        Parameters
        ----------
        x : tf.Tensor
            Input data
        sampling : bool
            Whether to sample parameter values from their variational
            distributions (if True, the default), or just use the
            Maximum a Posteriori parameter value estimates (if False).

        Returns
        -------
        preds : tf.Tensor of shape (Nsamples, 2)
            Output of this model, the predictions.  First column is
            the mean predictions, and second column is the standard
            deviation predictions.
        """

        # Pass data through core network
        x = self.core_net(x, sampling=sampling)
        x = tf.nn.relu(x)

        # Make predictions with each head network
        loc_preds = self.loc_net(x, sampling=sampling)
        std_preds = self.std_net(x, sampling=sampling)
        std_preds = tf.nn.softplus(std_preds)

        # Return mean and std predictions
        return tf.concat([loc_preds, std_preds], 1)

    def log_likelihood(self, x, y, sampling=True):
        """Compute the log likelihood of y given x"""

        # Compute mean and std predictions
        preds = self.call(x, sampling=sampling)

        # Return log likelihood of true data given predictions
        return tfd.Normal(preds[:, 0], preds[:, 1]).log_prob(y[:, 0])

    @tf.function
    def sample(self, x):
        """Draw one sample from the predictive distribution"""
        preds = self.call(x)
        return tfd.Normal(preds[:, 0], preds[:, 1]).sample()

    def samples(self, x, n_samples=1):
        """Draw multiple samples from the predictive distribution"""
        samples = np.zeros((x.shape[0], n_samples))
        for i in range(n_samples):
            samples[:, i] = self.sample(x)
        return samples

    @property
    def losses(self):
        """Sum of the KL divergences between priors + posteriors"""
        return (self.core_net.losses +
                self.loc_net.losses +
                self.std_net.losses)

D=2
# Batch size


# Learning rate
L_RATE = 1e-4

model2 = BayesianDensityNetwork([2,1],[1])

# Adam optimizer
optimizer = tf.keras.optimizers.Adam(lr=L_RATE)

N = x.shape[0]

@tf.function
def train_step(x_data, y_data):
    with tf.GradientTape() as tape:
        log_likelihoods = model2.log_likelihood(x_data, y_data)
        kl_loss = model2.losses
        elbo_loss = kl_loss/(N) - tf.reduce_mean(log_likelihoods)
    gradients = tape.gradient(elbo_loss, model2.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model2.trainable_variables))
    return elbo_loss

BATCH_SIZE = 2
EPOCHS = 1
# Fit the model

test_data=data['2015-12-20':'2015-12-22']
# new_data['RT_LMP_1']=new_data.RT_LMP.shift(1)
# new_data['DEMAND_1']=new_data.DEMAND.shift(1)
# new_data.drop(new_data.index[0])
from sklearn.preprocessing import MinMaxScaler
df=MinMaxScaler(feature_range=(0,10)).fit_transform(test_data[["DA_DEMD", "DA_LMP", "RT_LMP"]].dropna())
x_test= df[:,0:2]
y_test=df[:,2]
y=np.expand_dims(y,1)
y_test=np.expand_dims(y_test,1)
data_train = tf.data.Dataset.from_tensor_slices(
    (x, y)).shuffle(10000).batch(BATCH_SIZE)

# Make a TensorFlow Dataset from validation data
data_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(2)

elbo2 = []
mae2 = []
for i,epoch in enumerate(range(EPOCHS)):

    # Update weights each batch
    j=0
    for x_data, y_data in data_train:
        elbo2.append(train_step(x_data, y_data))
        j=j+1

    # Evaluate performance on validation data
    for x_data, y_data in data_test:
        y_pred = model2(x_data, sampling=False)[:, 0]
        mae2.append(mean_absolute_error(y_pred, y_data))

a=1

data_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(x_test.shape[0])

for x_data, y_data in data_test:
    samples2 = model2.samples(x_data, 1000)


def plot_output(m):
    fig3, axes3 = plt.subplots(5, 3, sharex='none', sharey='none')
    fig3.set_size_inches(6.4, 8)
    for i in range(5):
        for j in range(3):
            ix = i * 3 + j
            pred_dist = samples2[m - ix - 1, :]
            sns.distplot(pred_dist,  hist=True, color="r", ax=axes3[i][j])
            # sns.kdeplot(pred_dist, shade=True, ax=axes3[i][j])
            axes3[i][j].axvline(x=y_test[ix])

    axes3[4][0].set_ylabel('p(RT price| (DA price,DA demand))')
    axes3[1][0].set_ylabel('p(RT price| (DA price,DA demand))')
    axes3[4][0].set_xlabel('RT price')
    axes3[4][1].set_xlabel('RT price')
    axes3[4][2].set_xlabel('RT price')
    plt.show()


b=1
class BayesianDenseRegression(tf.keras.Model):
    """A multilayer fully-connected Bayesian neural network regression

    Parameters
    ----------
    dims : List[int]
        List of units in each layer
    name : str
        Name for the network

    Attributes
    ----------
    losses : tensorflow.Tensor
        Sum of the Kullback–Leibler divergences between
        the posterior distributions and their priors,
        over all layers in the network

    Methods
    -------
    call : tensorflow.Tensor
        Perform the forward pass of the data through
        the network, predicting both means and stds
    log_likelihood : tensorflow.Tensor
        Compute the log likelihood of y given x
    samples : tensorflow.Tensor
        Draw multiple samples from the predictive distribution
    """

    def __init__(self, dims, name=None):

        super(BayesianDenseRegression, self).__init__(name=name)

        # Multilayer fully-connected neural network to predict mean
        self.loc_net = BayesianDenseNetwork(dims)

        # Variational distribution variables for observation error
        self.std_alpha = tf.Variable([10.0], name='std_alpha')
        self.std_beta = tf.Variable([10.0], name='std_beta')

    def call(self, x, sampling=True):
        """Perform the forward pass, predicting both means and stds"""

        # Predict means
        loc_preds = self.loc_net(x, sampling=sampling)

        # Predict std deviation
        posterior = tfd.Gamma(self.std_alpha, self.std_beta)
        transform = lambda x: tf.sqrt(tf.math.reciprocal(x))
        N = x.shape[0]
        if sampling:
            std_preds = transform(posterior.sample([N]))
        else:
            std_preds = tf.ones([N, 1]) * transform(posterior.mean())

        # Return mean and std predictions
        return tf.concat([loc_preds, std_preds], 1)

    def log_likelihood(self, x, y, sampling=True):
        """Compute the log likelihood of y given x"""

        # Compute mean and std predictions
        preds = self.call(x, sampling=sampling)

        # Return log likelihood of true data given predictions
        return tfd.Normal(preds[:, 0], preds[:, 1]).log_prob(y[:, 0])

    @tf.function
    def sample(self, x):
        """Draw one sample from the predictive distribution"""
        preds = self.call(x)
        return tfd.Normal(preds[:, 0], preds[:, 1]).sample()

    def samples(self, x, n_samples=1):
        """Draw multiple samples from the predictive distribution"""
        samples = np.zeros((x.shape[0], n_samples))
        for i in range(n_samples):
            samples[:, i] = self.sample(x)
        return samples

    @property
    def losses(self):
        """Sum of the KL divergences between priors + posteriors"""

        # Loss due to network weights
        net_loss = self.loc_net.losses

        # Loss due to std deviation parameter
        posterior = tfd.Gamma(self.std_alpha, self.std_beta)
        prior = tfd.Gamma(10.0, 10.0)
        std_loss = tfd.kl_divergence(posterior, prior)

        # Return the sum of both
        return net_loss + std_loss