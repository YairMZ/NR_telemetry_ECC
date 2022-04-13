from scipy.stats import bernoulli
import numpy as np
import pandas as pd

rng = np.random.default_rng()

N_MLE = 50
p1_vec = np.arange(0.01, 0.99, 0.02)
f_vec = np.arange(0, 0.07, 0.01)

results = []

for p1 in p1_vec:
    for f in f_vec:
        x_n = bernoulli(p1)
        noise = bernoulli(f)
        l = []
        for _ in range(1000):
            x = x_n.rvs(size=N_MLE, random_state=rng)
            w = noise.rvs(size=N_MLE, random_state=rng)
            x = np.mod(x + w, 2)
            l.append(np.mean(x))
        v_hat = np.array(l)
        ignorant_error = np.abs(v_hat-p1)
        # print("maximal ignorant error: ", max(ignorant_error))
        # print("MSE of ignorant error: ", np.power(ignorant_error, 2).mean())
        p1_hat = np.clip((v_hat-f)/(1-2*f), 0, 1)
        ml_error = np.abs(p1_hat-p1)
        # max_error = max(ml_error)
        # MSE = np.power(ml_error, 2).mean()
        # print("maximal estimation error: ", max_error)
        # print("MSE: ", MSE)
        results.append({"p1": p1, "f": f, "error": ml_error, "ignorant_error": ignorant_error,
                        "ignorant_max_error": max(ignorant_error), "ignorant_mse": np.power(ignorant_error, 2).mean(),
                        "max_error": max(ml_error), "mse": np.power(ml_error, 2).mean()})

data = pd.DataFrame(results)

#%%
import plotly.express as px
mse_data = data[[]]
fig = px.density_contour(data, x="p1", y="f", z="mse")
fig.show()
#%%
f_vec = np.arange(0, 0.1, 0.005)
#%%
