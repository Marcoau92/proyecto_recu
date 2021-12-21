import numpy as np
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist

def model(x, y=None):
    teta = []
    teta0 = dist.Normal(loc=np.zeros(3), scale=5*np.ones(3)).to_event(1)
    teta.append(teta0)
    teta1 = dist.Normal(loc=np.zeros(3), scale=5*np.ones(3)).to_event(1)
    teta.append(teta1)
    teta2 = dist.Normal(loc=np.zeros(3), scale=5*np.ones(3)).to_event(1)
    teta.append(teta2)
    print(teta)
    lamda = []
    for i in range(197):
        j = 1
        temp = 0
        for j in range(3):
            temp = teta[j] + x[i,j]
        lamdai = jnp.exp(teta[0] + temp)
        lamda.append(lamdai)
    p = []
    for k in range(197):
        aux = (lamda[i]**y[i])/np.math.factorial(y[i])
        pk = aux*jnp.exp(-lamda[i])
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

    
    sampler = numpyro.infer.MCMC(sampler=numpyro.infer.NUTS(partial_model), 
                             num_samples=1000, num_warmup=100, thinning=1,
                             num_chains=2)

    sampler.run(rng_key_, x, y)

    sampler.print_summary(prob=0.9)
    
    return sampler.get_samples()
