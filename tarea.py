import numpy as np
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist

def model(x, y=None):
    prior_dist = dist.Normal(loc=jnp.zeros(4), scale=5*jnp.ones(4)).to_event(1) 
    theta = rutina_theta0(prior_dist)
    s_eps = numpyro.sample("s", dist.HalfCauchy(scale=5.0))
    with numpyro.plate('datos', size=197):
        f = numpyro.deterministic('lambda',
                            value=jnp.exp(theta[0] + theta[1]*x[0] + theta[2]*x[1] + theta[3]*x[2]))
        numpyro.sample("y", dist.Poisson(f), obs=y)
        return f

def run_mcmc_nuts(partial_model, x, y, rng_key_):
    """
    Implemente una función que calcula y retorna la traza de los parámetros del modelo. 
    Utilice el algoritmo de muestreo No U-turn (NUTS)
    Nota: Puede agregar argumentos a la función si lo necesita
    """
    rng_key, rngkey = random.split(rng_key)

    sampler = numpyro.infer.MCMC(sampler=numpyro.infer.NUTS(tarea.model), 
                             num_samples=1000, num_warmup=100, thinning=1,
                             num_chains=2)

    sampler.run(rngkey, x, y)



    sampler.print_summary(prob=0.9)
    
    return sampler.get_samples()

def rutina_theta0(prior_dist):
    theta = numpyro.sample("theta", prior_dist)
    return theta
def rutina_theta1(y,s_eps):
    theta01 = numpyro.sample('theta1',dist.Normal(loc=jnp.zeros(3), scale=s_eps*jnp.ones(3)),obs=1)
    return theta01
def rutina_theta2(y,s_eps):
    theta02 = numpyro.sample('theta2',dist.Normal(loc=jnp.zeros(3), scale=s_eps*jnp.ones(3)),obs=1)
    return theta02
def rutina_theta3(y,s_eps):
    theta03 = numpyro.sample('theta3',dist.Normal(loc=jnp.zeros(3), scale=s_eps*jnp.ones(3)),obs=1)
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
