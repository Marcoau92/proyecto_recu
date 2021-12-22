import numpy as np
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from decimal import Decimal
import scipy

def model(x, y=None):
    s = rutina_teta0(y)
    d = rutina_teta1(y)
    f = rutina_teta2(y)
    teta = np.array([s,d,f]).astype(np.float)
    print(teta)
    lamda = []
    for i in range(197):
        temp0 = 0
        temp1 = 0
        temp2 = 0
        temp0 = teta[0] + x[i,0]
        temp1 = teta[1] + x[i,1]
        temp2 = teta[2] + x[i,2]
        lamdai = np.exp(teta[0] + temp0 + temp1 + temp2)
        lamda.append(lamdai)
    p = []
    for k in range(197):
        #aux0 = Decimal((lamda[k]**y[k]))
        #aux1 = Decimal(np.math.factorial(y[k]))
        #print(y[k])
        #print(lamda[k])
        #print(aux0)
        #print(aux1)
        #aux = aux0/aux1
        #pk = Decimal(aux*jnp.exp(-lamda[k]))
        
        p.append(pk)
    return p

def run_mcmc_nuts(partial_model, x, y, rng_key_):
    """
    Implemente una función que calcula y retorna la traza de los parámetros del modelo. 
    Utilice el algoritmo de muestreo No U-turn (NUTS)
    Nota: Puede agregar argumentos a la función si lo necesita
    """
    rng_key = random.PRNGKey(1234)
    rng_key, rng_key_ = random.split(rng_key)
    print(rng_key_)
    
    sampler = numpyro.infer.MCMC(sampler=numpyro.infer.NUTS(partial_model), 
                             num_samples=1000, num_warmup=100, thinning=1,
                             num_chains=3)

    sampler.run(rng_key_, x, y)

    sampler.print_summary(prob=0.9)
    
    return sampler.get_samples()

def rutina_teta0(y):
    teta00 = dist.Normal(loc=jnp.zeros(3), scale=5*jnp.ones(3)).to_event(1)
    teta10 = numpyro.sample("teta00", dist.Poisson(teta00),obs= 1)
    return teta10
def rutina_teta1(y):
    teta01 = dist.Normal(loc=jnp.zeros(3), scale=5*jnp.ones(3)).to_event(1)
    teta11 = numpyro.sample("teta01", dist.Poisson(teta01),obs= 1)
    return teta11
def rutina_teta2(y):
    teta02 = dist.Normal(loc=jnp.zeros(3), scale=5*jnp.ones(3)).to_event(1)
    teta12 = numpyro.sample("teta02", dist.Poisson(teta02),obs= 1)
    return teta12
