import numpy as np
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
import random as random1
def model(x, y=None):
    prior_dist = dist.Normal(loc=jnp.zeros(4), scale=5*jnp.ones(4)).to_event(1) 
    #theta = [rutina_theta0(x),rutina_theta1(x),rutina_theta2(x),rutina_theta3(x)]
    db_w = dist.Normal(loc=0, scale=5)
    bw = numpyro.sample("bw", db_w)
    db_x = dist.Normal(loc=0, scale=5)
    bx = numpyro.sample("bx", db_x)
    db_y = dist.Normal(loc=0, scale=5)
    by = numpyro.sample("by", db_y)
    db_z = dist.Normal(loc=0, scale=5)
    bz = numpyro.sample("bz", db_z)
    theta = [bw,bx,by,bz]
    s_eps = numpyro.sample("s", dist.HalfCauchy(scale=5.0))
    with numpyro.plate('datos', size=197):
        f = numpyro.deterministic('lambda',
                            value=jnp.exp(theta[0] + jnp.dot(theta[1],x[0]) + jnp.dot(theta[2],x[1]) + jnp.dot(theta[3],x[2])))
        numpyro.sample("y", dist.Poisson(f), obs=y)
        return f

def run_mcmc_nuts(partial_model, x, y, rngkey):
    """
    Implemente una función que calcula y retorna la traza de los parámetros del modelo. 
    Utilice el algoritmo de muestreo No U-turn (NUTS)
    Nota: Puede agregar argumentos a la función si lo necesita
    """
    sampler = numpyro.infer.MCMC(sampler=numpyro.infer.NUTS(partial_model), 
                             num_samples=1000, num_warmup=100, thinning=1,
                             num_chains=3)
    sampler.run(rngkey, x, y)
    sampler.print_summary(prob=0.9)
    return sampler
def run_mcmc_BarkerMH(partial_model, x, y, rngkey):
    sampler = numpyro.infer.MCMC(sampler=numpyro.infer.BarkerMH(partial_model), 
                             num_samples=5000, num_warmup=100, thinning=1,
                             num_chains=3)
    sampler.run(rngkey, x, y)
    sampler.print_summary(prob=0.9)
    return sampler
def rutina_theta0(x):
    theta = numpyro.sample('theta0',dist.Normal(loc=0, scale=5),obs=1)
    return theta
def rutina_theta1(x):
    theta01 = numpyro.sample('theta1',dist.Normal(loc=0, scale=5),obs=1)
    return theta01
def rutina_theta2(x):
    theta02 = numpyro.sample('theta2',dist.Normal(loc=0, scale=5),obs=1)
    return theta02
def rutina_theta3(x):
    theta03 = numpyro.sample('theta3',dist.Normal(loc=0, scale=5),obs=1)
    return theta03
def plot_data(ax, x, y):
    ax.scatter(x[y == 0, 0], x[y == 0, 1], c='k', marker='o')
    ax.scatter(x[y == 1, 0], x[y == 1, 1], c='k', marker='x')
    ax.set_xlabel('x[:, 0]')
    ax.set_ylabel('x[:, 1]')
    
def get_predictive_samples(partial_model,
                           x_test,
                           rng_key_,
                           mcmc_trace=None,
                           num_samples=None):
    """
    Retorna las muestras de la distribución predictiva
    """
    if mcmc_trace is not None:
        predictive = numpyro.infer.Predictive(partial_model,
                                              posterior_samples=mcmc_trace)
    elif num_samples is not None:
        predictive = numpyro.infer.Predictive(partial_model,
                                              num_samples=num_samples)
    else:
        raise ValueError("Debe entregarse mcmc_trace o num_samples")
    return predictive(rng_key_, x_test)

def plot_predictive_posterior(ax, x1, x2, y, cmap, vmin=0, vmax=1, title=None):
    cmap = ax.pcolormesh(x1,
                         x2,
                         y.reshape(len(x1), len(x2)),
                         cmap=cmap,
                         shading='gouraud',
                         vmin=vmin,
                         vmax=vmax)
    plt.colorbar(cmap, ax=ax)
    if title is not None:
        ax.set_title(title)
        
def binary_mode(y):
    p = np.mean(y, axis=0)
    return p > 0.5


def entropy(y):
    p = np.mean(y, axis=0)
    return -p * np.log(p + 1e-10)

def plot_data(ax, x, y):
    ax.scatter(x[y == 0, 0], x[y == 0, 1], c='k', marker='o')
    ax.scatter(x[y == 1, 0], x[y == 1, 1], c='k', marker='x')
    ax.set_xlabel('x[:, 0]')
    ax.set_ylabel('x[:, 1]')
    
def autocorrelation(trace):
    """
    Retorna la autocorrelación de una traza
    """
    trace_norm = (trace - np.mean(trace)) / np.std(trace)
    rho = np.correlate(trace_norm, trace_norm, mode='full')
    return rho[len(rho) // 2:] / len(trace_norm)
