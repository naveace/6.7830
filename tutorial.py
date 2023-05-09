# %%
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from ppca import PPCA

def create_generative_model(N: int, W: np.ndarray, sigma_sq: float) -> go.Figure:
    assert isinstance(W, np.ndarray) and W.ndim == 2, f'W must be a 2 dimensional numpy array'
    assert W.shape == (2, 1), f'Currently we only support X 2 dimensional and Z 1 dimensional'
    assert isinstance(N, int) and N > 0, f'N must be a positive integer'
    assert isinstance(sigma_sq, int) or isinstance(sigma_sq, float) and sigma_sq >= 0, f'sigma must be a non-negative float'

    zs = np.random.randn(N)
    Xs = W @ zs.reshape(1, -1) + np.random.randn(2, N) * sigma_sq

    df = pd.DataFrame({"x": Xs[0], "y": Xs[1], "z": zs})

    fig = make_subplots(
        2,
        1,
        vertical_spacing=0.2,
        specs=[[{"type": "scatter"}], [{"type": "histogram"}]],
        row_titles=["Data Space", "Latent Space"],
    )
    fig.add_trace(
        go.Scatter(x=df.loc[:2]["x"], y=df.loc[:2]["y"], mode="markers"), row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=df.loc[:2]["z"], histnorm="probability"),
        row=2,
        col=1,
    )
    frames = [
        go.Frame(
            data=[
                go.Scatter(
                    x=df.loc[:k]["x"],
                    y=df.loc[:k]["y"],
                    mode="markers",
                ),
                go.Histogram(x=df.loc[:k]["z"], histnorm="probability"),
            ],
            name=f"frame{k}",
            traces=[0, 1],
        )
        for k in range(2, len(df))
    ]
    fig.update(frames=frames)


    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }


    fr_duration = 1  # customize this frame duration according to your data!!!!!
    sliders = [
        {
            "steps": [
                {
                    "args": [[f.name], frame_args(fr_duration)],
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]


    fig.update_layout(
        sliders=sliders,
    )
    fig.update_layout(
        # width=650,
        # height=400,
        showlegend=False,
        hovermode="closest",
    )
    return fig

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

def ppca_dataspace_EM(N: int, W: np.ndarray, sigma_sq: float, N_steps: int) -> go.Figure:
    assert isinstance(W, np.ndarray) and W.ndim == 2, f'W must be a 2 dimensional numpy array'
    assert W.shape == (2, 1), f'Currently we only support X 2 dimensional and Z 1 dimensional'
    assert isinstance(N, int) and N > 0, f'N must be a positive integer'
    assert isinstance(N_steps, int) and N_steps > 0, f'N_steps must be a positive integer'
    assert isinstance(sigma_sq, int) or isinstance(sigma_sq, float) and sigma_sq >= 0, f'sigma must be a non-negative float'

    ppca = PPCA(2, 1, N, W, sigma_sq)
    figures = []
    for _ in range(N_steps):
        figures.append(get_figure(ppca))
        ppca.E_step()
        ppca.M_step()
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
    return animated_figure

def get_MNIST_example(digit_number: int, M: int) -> go.Figure:
    from torchvision.datasets import EMNIST
    import numpy as np
    from pathlib import Path
    MNIST = EMNIST(root="data", split="mnist", train=False, download=not Path('data').exists())
    # Extrac tthe X and Y
    X = MNIST.data
    y = MNIST.targets

    mask = y == digit_number
    filtered_X = X.view(-1, 28 * 28)[mask].float()

    mean = filtered_X.mean(dim=0).numpy()
    cov_matrix = filtered_X.T.cov().numpy()
    filtered_X = filtered_X.numpy()

    # Compute the eigenvectors and eigenvalues
    eigenvectors, eigenvalues, _ = np.linalg.svd(cov_matrix, full_matrices=False)
    sigma_sq = np.sum(eigenvalues[M:]) / (28 * 28 - M)
    W = eigenvectors[:, :M] @ np.sqrt(np.diag(eigenvalues[:M]) - sigma_sq * np.eye(M))

    C = W @ W.T + sigma_sq * np.eye(28 * 28)
    scores = np.diag(-1 * (filtered_X - mean) @ np.linalg.inv(C) @ (filtered_X - mean).T)
    sorted_likelihoods = np.argsort(scores)

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    fig = make_subplots(
        2,
        6,
        vertical_spacing=0.2,
        horizontal_spacing=0.1,
        row_titles=["Least Likely", "Most Likely"],
    )
    process = lambda x: np.rot90(x.reshape(28, 28), 1)
    for i in range(6):
        fig.add_trace(
            go.Heatmap(
                z=process(filtered_X[sorted_likelihoods[i]]),
                colorscale="gray",
                showscale=False,
            ),
            row=1,
            col=i + 1,
        )
        fig.add_trace(
            go.Heatmap(
                z=process(filtered_X[sorted_likelihoods[-i - 1]]),
                colorscale="gray",
                showscale=False,
            ),
            row=2,
            col=i + 1,
        )
    # Turn off all the axes
    for i in range(1, 3):
        for j in range(1, 7):
            fig.update_xaxes(showticklabels=False, row=i, col=j)
            fig.update_yaxes(showticklabels=False, row=i, col=j)

    return fig