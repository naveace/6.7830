#%%
%load_ext autoreload 
%autoreload 2 
import numpy as np
from ppca import PPCA
#%%
np.random.seed(9)
p = PPCA(2, 1, 3, np.array([1, 0]).reshape(2, 1), 1)
fig = p.plot_EM(30)
fig.update_layout(title="Distance from True Params + ELBO")


# %%
np.random.seed(5)
p = PPCA(2, 1, 2, np.array([1, 0]).reshape(2, 1), 1)
fig = p.plot_EM(100)
fig.update_layout(title="Distance from True Params + ELBO")

# %%
p = PPCA(2, 1, 10_000, np.array([2, 0]).reshape(2, 1), 0.1)
p.plot_EM(10)
# %%