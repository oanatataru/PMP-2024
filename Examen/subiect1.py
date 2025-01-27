import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

#a)

data = pd.read_csv("date_alegeri_turul2.csv")
X = data[["Varsta", "Sex", "Educatie", "Venit"]]
y = data["Vot"]

with pm.Model() as model_a:
    beta = pm.Normal("beta", mu=0, sigma=10, shape=X.shape[1])
    alfa = pm.Normal("alfa", mu=0, sigma=10)

    p = pm.math.sigmoid(alfa + pm.math.dot(X, beta))

    trace_a = pm.sample(1000, return_inferencedata=True)

#b)

summary_a = az.summary(trace_a, var_names=["beta"])
print(summary_a)

top_2_vars = np.abs(summary_a["mean"]).sort_values(ascending=False).index[:2]
print(f"cele mai importante 2 variabile: {top_2_vars}")

#c)

X_reduced = X[top_2_vars] #model cu cele 2 variabile

with pm.Model() as model_c:
    beta = pm.Normal("beta", mu=0, sigma=10, shape=X_reduced.shape[1])
    alfa = pm.Normal("alfa", mu=0, sigma=10)

    p = pm.math.sigmoid(alfa + pm.math.dot(X_reduced, beta))

    trace_c = pm.sample(1000, return_inferencedata=True)

x_min, x_max = X_reduced.iloc[:, 0].min(), X_reduced.iloc[:, 0].max()
y_min, y_max = X_reduced.iloc[:, 1].min(), X_reduced.iloc[:, 1].max()

plt.scatter(X_reduced.iloc[:, 0], X_reduced.iloc[:, 1], c=y, label="Votanți")
plt.title("granita de decizie (cu 2 variabile)")
plt.xlabel(top_2_vars[0])
plt.ylabel(top_2_vars[1])
plt.show()

#d

waic_a = az.waic(trace_a, model_a) #modelul de la a
waic_c = az.waic(trace_c, model_c) #modelul de la c

loo_a = az.loo(trace_a, model_a)
loo_c = az.loo(trace_c, model_c)

print(f"{waic_a.waic}")
print(f"{waic_c.waic}")
print(f"{loo_a.loo}")
print(f"{loo_c.loo}")