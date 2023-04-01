#%%
from typing import Dict
from tqdm.auto import tqdm
import numpy as np
import plotly.express as px
import pandas as pd

#%%
# D = 2
# M = 2
# N = 1_000
# true_Z = np.random.randn(N, M)
# true_W = np.eye(D, M)
# true_sigma_sq = 1
# X =  true_Z @ true_W.T + np.random.randn(N, D) * np.sqrt(true_sigma_sq)
# # Plot the data
# fig = px.scatter(
#     x=X[:, 0],
#     y=X[:, 1],
#     title='Data',
# )
# fig.show()
# %%
# W = np.eye(D, M)
# bar_x: np.ndarray = np.mean(X, axis=0).reshape(-1, 1)
# sigma_sq = 1
# def bf_M() -> np.ndarray:
#     global W
#     return W.T @ W + sigma_sq * np.eye(M)
# def bf_E_z(n: int) -> np.ndarray:
#     global W
#     out = np.linalg.inv(bf_M()) @ W.T @ (X[n, :].reshape(-1, 1) - bar_x)
#     assert out.shape == (M, 1)
#     return out
# def bf_E_zzT(n: int) -> np.ndarray:
#     global W, sigma_sq
#     out = sigma_sq * np.linalg.inv(bf_M()) + bf_E_z(n) @ bf_E_z(n).T
#     assert out.shape == (M, M)
#     return out

# def W_new() -> np.ndarray:
#     x_z_outer_product = np.zeros((D, M))
#     for n in range(N):
#         addition = (X[n, :].reshape(-1, 1) - bar_x) @ bf_E_z(n).T
#         assert addition.shape == (D, M)
#         x_z_outer_product += addition
#     z_z_outer_product = np.zeros((M, M))
#     for n in range(N):
#         addition = bf_E_zzT(n)
#         assert addition.shape == (M, M)
#         z_z_outer_product += addition
#     return x_z_outer_product @ np.linalg.inv(z_z_outer_product)

# def sigma_sq_new() -> float:
#     W_new_ = W_new()
#     out = 0
#     for n in range(N):
#         out += np.linalg.norm(X[n, :].reshape(-1, 1) - bar_x) ** 2
#         out -= (2 * bf_E_z(n).T @ W_new_.T @ (X[n, :].reshape(-1, 1) - bar_x))[0, 0]
#         out += np.trace(bf_E_zzT(n) @ W_new_.T @ W_new_)
#     return out / (N * D)

# def update() -> Dict[str, float]:
#     global W, sigma_sq
#     tmp_W = W_new()
#     tmp_sigma_sq = sigma_sq_new()
#     W = tmp_W
#     sigma_sq = tmp_sigma_sq
#     return {
#         'Distance from true W': np.linalg.norm(W - true_W),
#         'Distance from true sigma_sq': np.abs(true_sigma_sq - sigma_sq),
#     }
# from tqdm import tqdm
# out = [
#     update()
#     for _ in tqdm(range(20))
# ]
# %%
# out = pd.DataFrame(out)
# fig = px.line(
#     out,
#     y=['Distance from true W', 'Distance from true sigma_sq'],
#     title='Distance from true parameters',
# )
# fig.show()
# %%
class PPCA:
    def __init__(
        self, D: int, M: int, N: int, W: np.ndarray, sigma_sq: float
    ) -> None:
        self.D = D
        self.M = M
        self.N = N
        self.W = W
        self.sigma_sq = sigma_sq
        self.Z = np.random.randn(N, M)
        self.X: np.ndarray = self.Z @ self.W.T + np.random.randn(N, D) * np.sqrt(
            self.sigma_sq
        )
        self.bar_x = self.X.mean(axis=0).reshape(D, 1)
        self.exp_z_hat = np.empty((N, M))
        self.exp_z_zT_hat = np.empty((N, M, M))
        self.W_hat = np.random.randn(D, M)
        self.sigma_sq_hat: float = 1.0

    def E_step(self):
        bf_M = self.W_hat.T @ self.W_hat + self.sigma_sq_hat * np.eye(self.M)
        # TODO: Vectorize
        for n in range(self.N):
            self.exp_z_hat[n] = (
                np.linalg.inv(bf_M)
                @ self.W_hat.T
                @ (self.X[n, :].reshape(self.D, 1) - self.bar_x)
            ).reshape(1, self.M)
            exp_z_hat_n = self.exp_z_hat[n].reshape(self.M, 1)
            self.exp_z_zT_hat[n] = (
                self.sigma_sq_hat * np.linalg.inv(bf_M) + exp_z_hat_n @ exp_z_hat_n.T
            )

    def M_step(self):
        W_new = np.zeros_like(self.W_hat)
        left_addition = np.zeros((self.D, self.M))
        for n in range(self.N):
            E_z_n = self.exp_z_hat[n].reshape(self.M, 1)
            x_n = self.X[n].reshape(self.D, 1)
            left_addition += (x_n - self.bar_x) @ E_z_n.T
        W_new = left_addition @ np.linalg.inv(self.exp_z_zT_hat.sum(axis=0))
        assert W_new.shape == (self.D, self.M)
        sigma_sq_new = 0
        for n in range(self.N):
            x_n = self.X[n].reshape(self.D, 1)
            E_z_n = self.exp_z_hat[n].reshape(self.M, 1)
            addition = (
                (x_n - self.bar_x).T @ (x_n - self.bar_x)
                - 2 * E_z_n.T @ W_new.T @ (x_n - self.bar_x)
                + np.trace(self.exp_z_zT_hat[n] @ W_new.T @ W_new)
            )
            assert addition.shape == (1, 1), addition.shape
            sigma_sq_new += addition[0, 0] / (self.N * self.D)
        self.W_hat, self.sigma_sq_hat = W_new, sigma_sq_new

    def EM(self, n_steps: int = 1) -> None:
        for n in tqdm(list(range(n_steps)), leave=False):
            self.E_step()
            self.M_step()

    def distances_from_truth(self) -> pd.Series:
        return pd.Series(
            {
                "Distance from true W": np.linalg.norm(self.W - self.W_hat),
                "Distance from true sigma_sq": np.abs(
                    self.sigma_sq - self.sigma_sq_hat
                ),
            }
        )

    def plot_EM(self, n_steps: int) -> None:
        df = []
        for n in range(n_steps):
            self.EM()
            df.append(self.distances_from_truth().to_frame().T)
        df = pd.concat(df, axis=0, ignore_index=True)
        fig = px.line(
            df,
            y=["Distance from true W", "Distance from true sigma_sq"],
            title="Distance from true parameters",
        )
        fig.show()
#%%
p = PPCA(2, 2, 10_000, np.array([[1, 0], [0, 1]]), 0.0)
bf_M = p.W.T @ p.W + p.sigma_sq * np.eye(2)
for n in range(10_000):
    p.exp_z_hat[n] = (
        np.linalg.inv(bf_M)
        @ p.W.T
        @ (p.X[n, :].reshape(2, 1) - p.bar_x)
    ).reshape(1, 2)
    exp_z_hat_n = p.exp_z_hat[n].reshape(2, 1)
    p.exp_z_zT_hat[n] = (
        p.sigma_sq * np.linalg.inv(bf_M) + exp_z_hat_n @ exp_z_hat_n.T
    )
p.M_step()
#%%
p.sigma_sq_hat
#%%
(p.W_hat @ p.W_hat.T)
# %%
out = []
for N in [100_000 ]:
    p = PPCA(2, 2, N, np.array([[10, 0], [0, 1]]), 0.0)
    p.EM(10)
    out.append(p.distances_from_truth().to_frame().T)
out = pd.concat(out, axis=0, ignore_index=True)
# %%
p.sigma_sq_hat
# %%
p.W_hat @ p.W_hat.T
# %%
np.linalg.eig(p.X.T @ p.X / p.N)
# %%
