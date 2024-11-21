import numpy as np
import scipy.stats as stats
import matplotlib.pylab as plt

#a)
moneda = ["s", "b"]
aruncari = ["s", "b", "b", "b", "s", "b", "s", "b", "b", "b"]
nr_aruncari = 10

alpha_prior = 1
beta_prior = 1

## determinare parametri distributie a posteriori

## grafic (histograma)
plt.hist(aruncari, bins=30, edgecolor='black', alpha=1)
plt.title('Distributie subpct a)')
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.show()

#b)

aruncari_1 = ["s", "s", "b", "s", "b", "b", "s", "b", "b", "s"]
nr_aruncari_1 = 10

#alpha_prior = 1
#beta_prior = 1 (consideram o ditributie beta uniforma, precum cea data initial la a)

#alpha_prior_a = (punem ca rezultate ce am obtinut anterior la
#beta_prior_b =

## determinare parametri distributie a posteriori

## grafic (histograma)
plt.hist(aruncari_1, bins=30, edgecolor='black', alpha=0.5) #punem la alphaul corespunzator noii distributii gasite la a)
plt.title('Distributie subpct b)')
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.show()

##comparam distributiile intre ele


## cu pymc, ne folosim de variabilele definite mai sus
#import pymc as pm

#a)
#with pm.Model() as model:
#    p = pm.Beta("p", alpha=1, beta=1, shape=10) #pt ca avem 10 aruncari
#    k_aruncari = pm.Binomial("k_obs", n=aruncari, p=p, observed=)
#    p_mean = pm.Deterministic("p_mean", pm.math.mean(p))
#    trace = pm.sample(10, chains=, target_accept=)