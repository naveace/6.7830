#%%
import numpy as np
from ppca import PPCA

# %%
np.random.seed(0)
p = PPCA(2, 1, 2, np.array([1, 0]).reshape(2, 1), 1)
fig = p.plot_EM(100)
fig.update_layout(title="Distance from True Params + ELBO")

#%%
np.random.seed(3)
p = PPCA(2, 1, 3, np.array([1, 0]).reshape(2, 1), 1)
fig = p.plot_EM(100)
fig.update_layout(title="Distance from True Params + ELBO")

# %%
