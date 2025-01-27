import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

data = pd.read_csv("iris.csv")
caracteristics = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = data[caracteristics].values

clustere = 3
models = {}

for caracteristic in caracteristics:
    caracteristics_data = data[caracteristic].values.reshape(-1, 1)

    gmm = GaussianMixture(n_components=clustere, random_state=42)
    gmm.fit(caracteristics_data)
    models[caracteristic] = gmm

    x = np.linspace(caracteristics_data.min() - 1, caracteristics_data.max() + 1, 1000).reshape(-1, 1)
    pdf = np.exp(gmm.score_samples(x))

    plt.hist(caracteristics_data, bins=30, density=True, alpha=0.5, label='date originale')
    plt.plot(x, pdf, label=f'densitate ({caracteristics})')
    plt.title(f"mixtura gaussiana - {caracteristics}")
    plt.xlabel("valoare")
    plt.ylabel("densitate")
    plt.show()

## verificare ce caracteristica separa mai bine datele

for feature, model in models.items():
    labels = model.predict(data[feature].values.reshape(-1, 1))
    cluster_sizes = np.unique(labels, return_counts=True)[1]
    print(f"caracteristica: {feature}, distribuție clustere: {cluster_sizes}")



