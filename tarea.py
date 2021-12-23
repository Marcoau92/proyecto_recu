import numpy as np
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist

def model(x, y=None):
    #np.random.seed(1234)
    s_eps = 5
    s = rutina_theta0(y,s_eps)
    #b_star, w_star, s_eps_star, N = 10, 2.5, 1., 10
    #x1 = np.random.randn(N)
    #y1 = b_star + w_star*x1 +  np.random.randn(N)*s_eps_star
    d = rutina_theta1(y,s_eps)
    f = rutina_theta2(y,s_eps)
    g = rutina_theta3(y,s_eps)
    theta = jnp.array([s,d,f,g]).astype(np.float)
    p = []
    
    for i in range(197):
        temp0 = theta[1] * x[i,0]
        temp1 = theta[2] * x[i,1]
        temp2 = theta[3] * x[i,2]
        lamdai = jnp.exp(theta[0] + temp0 + temp1 + temp2)
        p.append(lamdai)
    lamda = jnp.array(p)
    print(lamda.shape)
    print(y.shape)
    with numpyro.plate('datos', size=len(y)):
        f = numpyro.deterministic('lamda', value=lamda)
        print(f.shape)
        #numpyro.sample('c', dist.Poisson(f), obs=y)
        return f
    #sam = numpyro.sample('y', dist.Poisson(lamda), obs=y)
    #print(sam)
    #return sam

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

    sampler.run(rng_key_,x, y)

    sampler.print_summary(prob=0.9)
    
    return sampler.get_samples()

def rutina_theta0(y,s_eps):
    theta00 = numpyro.sample('theta0',dist.Normal(loc=jnp.zeros(3), scale=s_eps*jnp.ones(3)),obs=1)
    return theta00
def rutina_theta1(y,s_eps):
    theta01 = numpyro.sample('theta1',dist.Normal(loc=jnp.zeros(3), scale=s_eps*jnp.ones(3)),obs=1)
    return theta01
def rutina_theta2(y,s_eps):
    theta02 = numpyro.sample('theta2',dist.Normal(loc=jnp.zeros(3), scale=s_eps*jnp.ones(3)),obs=1)
    return theta02
def rutina_theta3(y,s_eps):
    theta03 = numpyro.sample('theta3',dist.Normal(loc=jnp.zeros(3), scale=s_eps*jnp.ones(3)),obs=1)
    return theta03
