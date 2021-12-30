import numpy as np
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
import random as random1
import matplotlib.pyplot as plt

def model(x, y=None):
    prior_dist = dist.Normal(loc=0, scale=5)
    theta0 = numpyro.sample("theta0", prior_dist)
    theta1 = numpyro.sample("theta1", prior_dist)
    theta2 = numpyro.sample("theta2", prior_dist)
    theta3 = numpyro.sample("theta3", prior_dist)
    theta = [theta0,theta1,theta2,theta3]
    print(theta)
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    print(x1)
    print(x2)
    print(x3)
    with numpyro.plate('datos', size=197) as ind:
        f = numpyro.deterministic('lambda',
                            value=jnp.exp(theta[0] + jnp.dot(theta[1],x1) + jnp.dot(theta[2],x2) + jnp.dot(theta[3],x3)))
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
                             num_samples=1000, num_warmup=100, thinning=1,
                             num_chains=3)
    sampler.run(rngkey, x, y)
    sampler.print_summary(prob=0.9)
    return sampler
    
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

def trazas(model):
    p=[]
    theta0 = np.array(model['theta0'])
    theta1 = np.array(model['theta1'])
    theta2 = np.array(model['theta2'])
    theta3 = np.array(model['theta3'])
    fig, ax = plt.subplots(2, 2, figsize=(16, 8), tight_layout=True)   
    fig.suptitle("Gráficas de Trazas")
    for a in model:
        if a == 'theta0':
            ax[0,0].set_xlabel('Iteraciones')
            ax[0,0].set_ylabel('Traza')
            ax[0,0].set_title(a)
            ax[0,0].plot(theta0)
        elif a == 'theta1':
            ax[0,1].set_xlabel('Iteraciones')
            ax[0,1].set_ylabel('Traza')
            ax[0,1].set_title(a)
            ax[0,1].plot(theta1)
        elif a == 'theta2':
            ax[1,0].set_xlabel('Iteraciones')
            ax[1,0].set_ylabel('Traza')
            ax[1,0].set_title(a)
            ax[1,0].plot(theta2)
        elif a == 'theta3':
            ax[1,1].set_xlabel('Iteraciones')
            ax[1,1].set_ylabel('Traza')
            ax[1,1].set_title(a)
            ax[1,1].plot(theta3)

            