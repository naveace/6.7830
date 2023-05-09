#%%
from ppca import PPCA
import numpy as np
import plotly.express as px
import pandas as pd

#%%
pc = np.array([1, 1]) / np.sqrt(2)
ppca = PPCA(2, 1, 1_000, pc.reshape(2, 1), 1.0)
df = pd.DataFrame(
    {
        "w_x": pd.Series([], dtype=float),
        "w_y": pd.Series([], dtype=float),
        "sigma_sq": pd.Series([], dtype=float),
        "ELBO": pd.Series([], dtype=float),
        "iteration": pd.Series([], dtype=int),
    }
)
#%%
import plotly.graph_objects as go
def get_figure(ppca: PPCA) -> go.Figure:
    normalized_W_hat = ppca.W_hat / np.linalg.norm(ppca.W_hat)
    normalized_W = ppca.W / np.linalg.norm(ppca.W)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=ppca.X[0, :],
                y=ppca.X[1, :],
                mode="markers",
                name="data"
            ),
            go.Scatter(
                x=[0, normalized_W_hat[0, 0]],
                y=[0, normalized_W_hat[1, 0]],
                mode="lines",
                name="Estimated W"
            ),
            go.Scatter(
                x=[0, normalized_W[0, 0]],
                y=[0, normalized_W[1, 0]],
                mode="lines",
                name="True W"
            )
        ],
        layout=go.Layout(
            xaxis_title="x",
            yaxis_title="y",
            title="Data and Estimated W",
        )
    )
    return fig

#%%
figures = []
for _ in range(30):
    figures.append(get_figure(ppca))
    ppca.E_step()
    ppca.M_step()

#%%

animated_figure = go.Figure()
for fig in figures:
    animated_figure.add_trace(fig.data[0])
    animated_figure.add_trace(fig.data[1])
    animated_figure.add_trace(fig.data[2])
animated_figure.update_traces(visible=False)
for i in range(3):
    animated_figure.data[i].visible = True
steps = []
for i in range(len(figures)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(animated_figure.data)}],
        label="Iteration {}".format(i),
    )
    step["args"][0]["visible"][3*i] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][3*i+1] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][3*i+2] = True  # Toggle i'th trace to "visible"
    steps.append(step)
sliders = [dict(active=0, pad={"t": 50}, steps=steps )]
animated_figure.update_layout(sliders=sliders)
animated_figure.show()

#%%
for i in range(1, 11):
    ppca.E_step()
    ppca.M_step()
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "w_x": pd.Series([ppca.W_hat[0, 0]], dtype=float),
                    "w_y": pd.Series([ppca.W_hat[1, 0]], dtype=float),
                    "sigma_sq": pd.Series([ppca.sigma_sq_hat], dtype=float),
                    "ELBO": pd.Series([ppca.ELBO()], dtype=float),
                    "iteration": pd.Series([i], dtype=int),
                }
            ),
        ],
        axis=0,
        ignore_index=True,
    )


# %%
df

# %%
fig = px.scatter(
    df,
    x="w_x",
    y="w_y",
    animation_frame="iteration",

)
fig.update_xaxes(range=[-1.95, 1.95])
fig.update_yaxes(range=[-1.95, 1.95])
fig.show()
# %%
