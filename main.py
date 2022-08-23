import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import numpy as onp
import jax
import jax.numpy as jnp
from jax import value_and_grad, grad, jit, random, lax

from jax.nn import log_softmax
from jax.scipy.special import logsumexp as lse


def init_model(rng, X, Z):
    rng_z, rng_x = random.split(rng)
    log_p_x_given_z = log_softmax(random.normal(rng_z, (X, Z)), 0)
    log_p_z = log_softmax(random.normal(rng_z, (Z,)))
    return log_p_x_given_z, log_p_z

seed = 1234
rng = random.PRNGKey(seed)

Z = 128
X = 56
n_trials = 2

def get_error(rng, n_samples):
    rng, rng_model = random.split(rng)
    log_p_x_given_z, log_p_z = init_model(rng_model, X, Z)

    rng, rng_obs = random.split(rng)
    observations = random.choice(rng_obs, X, shape=(n_trials,))
    log_p_z_samples, z_samples = lax.top_k(log_p_z, n_samples)


    def log_marginal(lpxgz, lpz, obs, z):
        return lse(lpxgz[obs] + lpz).mean()

    def log_approx_sum(lpxgz, lpz, obs, z):
        return lse(lpxgz[obs[:,None], z] + lpz[z]).mean()

    def elbo_prior(lpxgz, lpz, obs, z):
        return (
            lpxgz[obs[:,None], z]
            + lax.stop_gradient(lpxgz[obs[:,None], z] * lpz[z])
        ).sum(1).mean()

    def expected_complete(lpxgz, lpz, obs, z):
        lpxz = lpxgz + lpz
        lpzgx = lpxz - lse(lpxz, axis=1, keepdims=True)
        return (
            lax.stop_gradient(jnp.exp(lpzgx[obs[:,None],z]))
            * lpxz[obs[:,None], z]
        ).sum(1).mean()

    args = (log_p_x_given_z, log_p_z, observations, z_samples)

    log_approx_sum(*args)
    elbo_prior(*args)
    expected_complete(*args)

    log_marginal_value, log_marginal_grad = value_and_grad(log_marginal, argnums=(0,1))(*args)
    log_approx_sum_value, log_approx_sum_grad = value_and_grad(log_approx_sum, argnums=(0,1))(*args)
    elbo_prior_value, elbo_prior_grad = value_and_grad(elbo_prior, argnums=(0,1))(*args)
    ec_value, ec_grad = value_and_grad(expected_complete, argnums=(0,1))(*args)

    """
    print("sum squared error between ground truth and approx sum")
    print(jnp.square(log_marginal_grad[1] - log_approx_sum_grad[1]).sum())
    print("sum squared error between ground truth and elbo prior")
    print(jnp.square(log_marginal_grad[1] - elbo_prior_grad[1]).sum())
    print("sum squared error between ground truth and approx true")
    print(jnp.square(log_marginal_grad[1] - ec_grad[1]).sum())
    """
    e1 = jnp.square(log_marginal_grad[1] - log_approx_sum_grad[1]).sum().item()
    e2 = jnp.square(log_marginal_grad[1] - elbo_prior_grad[1]).sum().item()
    e3 = jnp.square(log_marginal_grad[1] - ec_grad[1]).sum().item()
    return e1, e2, e3

data = []
for n_samples in range(8, 129, 8):
    e1, e2, e3 = get_error(rng, n_samples)
    data.append((n_samples, e1, "approx-sum"))
    data.append((n_samples, e2, "elbo-prior"))
    data.append((n_samples, e3, "expected-complete"))


df = pd.DataFrame(data, columns = ["samples", "error", "grad"])
sns.lineplot(data=df, x="samples", y="error", hue="grad")
plt.savefig("grad_error_plot.png")
