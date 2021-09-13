# --- Import libraries --- #
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import pymc3 as pm
import arviz as az
import theano
import theano.tensor as tt

# --- Read data --- #
data = pd.read_csv('./data/machine_readable.csv')

# --- Standardise continuous values --- #
data_prep = data[['LogD', 'Cbrain/Cblood', 'Syn_EC30',
                  'Syn_Viability_EC30', 'NNF_EC50min', 'NNF_EC50max', 'NNF_LDH_EC50', 'NNF_AB_EC50']]
sc = StandardScaler()
data_scaled = sc.fit_transform(data_prep)
data_scaled = pd.DataFrame(data_scaled)
data_scaled.columns = ['LogD', 'Cbrain/Cblood', 'Syn_EC30',
                       'Syn_Viability_EC30', 'NNF_EC50min', 'NNF_EC50max', 'NNF_LDH_EC50', 'NNF_AB_EC50']

# Add other discrete values and details
discrete = data[['Chemical', 'CASRN', 'DNT', 'BBB', 'Pgp_inhibition', 'Pgp_substrate', 'Pgp_active',
                 'BDNF_Reduction', 'Activity_Syn', 'Activity_NNF']]
data = discrete.join(data_scaled, lsuffix="_left", rsuffix="_right")

# Print transformed data
data.head()

# --- Define predictors and outcomes --- #

# BDNF given by continuous and discrete variables
x_bdnf = pd.DataFrame(data[['LogD', 'Cbrain/Cblood', 'BBB',
                            'Pgp_inhibition', 'Pgp_substrate', 'Pgp_active']]).values
y_bdnf = pd.DataFrame(data[['BDNF_Reduction']]).values

# SYN
x_syn = pd.DataFrame(data[['Syn_EC30', 'Syn_Viability_EC30']]).values
Y_syn = pd.DataFrame(data[['Activity_Syn']]).values  # Y because no missing values

# NNF
x_nnf = pd.DataFrame(data[['NNF_EC50min', 'NNF_EC50max', 'NNF_LDH_EC50', 'NNF_AB_EC50']]).values
Y_nnf = pd.DataFrame(data[['Activity_NNF']]).values  # Y because no missing values

# x_dnt is given by causal relations BDNF->SYN->NNF->DNT<-BNDF
Y_dnt = pd.DataFrame(data[['DNT']]).values  # Y because no missing values

# --- Mask missing data --- #
# BDNF
x_bdnf_missing = np.isnan(x_bdnf)
X_bdnf_train = np.ma.masked_array(x_bdnf, mask=x_bdnf_missing)
y_bdnf_missing = np.isnan(y_bdnf)
Y_bdnf_train = np.ma.masked_array(y_bdnf, mask=y_bdnf_missing)

# SYN
x_syn_missing = np.isnan(x_syn)
X_syn_train = np.ma.masked_array(x_syn, mask=x_syn_missing)

# NNF
x_nnf_missing = np.isnan(x_nnf)
X_nnf_train = np.ma.masked_array(x_nnf, mask=x_nnf_missing)

# --- Define and fit the model --- #
with pm.Model() as model_hierar:
    # Define hyperpriors
    mu_beta = pm.Normal('mu_beta', mu=0, sd=0.01)
    sd_beta = pm.HalfNormal('sd_beta', sd=1)
    # Define priors
    beta_bdnf = pm.Normal('beta_bdnf', mu=mu_beta, sd=sd_beta, shape=(6, 1))
    beta_syn = pm.Normal('beta_syn', mu=mu_beta, sd=sd_beta, shape=(2, 1))
    beta_nnf = pm.Normal('beta_nnf', mu=mu_beta, sd=sd_beta, shape=(4, 1))

    # Imputation of X missing values for BDNF
    Xmu_bdnf = pm.Normal('Xmu_bdnf', mu=0, sd=0.01, shape=(1, 6))
    Xsigma_bdnf = pm.HalfNormal('Xsigma_bdnf', sd=1, shape=(1, 6))
    X_bdnf_modelled = pm.Normal('X_bdnf_modelled',
                                mu=Xmu_bdnf, sigma=Xsigma_bdnf, observed=X_bdnf_train)

    # Likelihood for BDNF
    # SLogP, Cbrain/Cblood, BBB, Pgp->BDNF
    lp_bdnf = pm.Deterministic('lp_bdnf', pm.math.dot(X_bdnf_modelled, beta_bdnf))
    y_obs_bdnf = pm.Bernoulli('y_obs_bdnf', logit_p=lp_bdnf, observed=Y_bdnf_train)

    # Imputation of X missing values for SYN
    Xmu_syn = pm.Normal('Xmu_syn', mu=0, sd=0.01, shape=(1, 2))
    Xsigma_syn = pm.HalfNormal('Xsigma_syn', sd=1, shape=(1, 2))
    X_syn_modelled = pm.Normal('X_syn_modelled',
                               mu=Xmu_syn, sigma=Xsigma_syn, observed=X_syn_train)

    # Likelihood for SYN
    # BDNF->SYN
    lp_syn = pm.Deterministic('lp_syn', lp_bdnf + pm.math.dot(X_syn_modelled, beta_syn))
    y_obs_syn = pm.Bernoulli("y_obs_syn", logit_p=lp_syn, observed=Y_syn)

    # Imputation of X missing values for NNF
    Xmu_nnf = pm.Normal('Xmu_nnf', mu=0, sd=0.01, shape=(1, 4))
    Xsigma_nnf = pm.HalfNormal('Xsigma_nnf', sd=1, shape=(1, 4))
    X_nnf_modelled = pm.Normal('X_nnf_modelled',
                               mu=Xmu_nnf, sd=Xsigma_nnf, observed=X_nnf_train)

    # Likelihood for NNF
    # BDNF->SYN->NNF
    lp_nnf = pm.Deterministic('lp_nnf', lp_syn + pm.math.dot(X_nnf_modelled, beta_nnf))
    y_obs_nnf = pm.Bernoulli("y_obs_nnf", logit_p=lp_nnf, observed=Y_nnf)

    # Define causal relationships for DNT
    lp_dnt = pm.Deterministic('lp_dnt', lp_bdnf + lp_syn + lp_nnf)
    y_obs_dnt = pm.Bernoulli('y_obs_dnt', logit_p=lp_dnt, observed=Y_dnt)

# --- Checking the proposed structure of model --- #
model_hierar.check_test_point()

# --- Run inferences and compute posterior distributions --- #
with model_hierar:
    trace_hierar = pm.sample(cores=1, nuts={'target_accept': 0.90})
    # Predictions
    posterior_hierar = pm.sample_posterior_predictive(trace_hierar)

# --- Convert and store traces in Arviz format --- #
# idata_hierar = az.from_pymc3(trace=trace_hierar,
#                              posterior_predictive=posterior_hierar,
#                              model=model_hierar)
# idata_hierar.to_netcdf("../data/idata_hierar.nc")
