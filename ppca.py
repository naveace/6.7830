#%%
from typing import Dict
from numpy.linalg import inv
from tqdm.auto import tqdm
import numpy as np
import plotly.express as px
from plotly.graph_objects import Figure
import pandas as pd

#%%
class PPCA:
    def __init__(self, D: int, M: int, N: int, W: np.ndarray, sigma_sq: float) -> None:
        self.D = D
        self.M = M
        self.N = N
        self.W = W  # D X M
        self.sigma_sq = sigma_sq
        self.Z = np.random.randn(M, N)  # Each column is a sample in all that follows
        self.X: np.ndarray = (self.W @ self.Z) + np.random.randn(D, N) * np.sqrt(
            self.sigma_sq
        )
        self.bar_x =  self.X.mean(axis=1).reshape(D, 1)
        self.exp_z_hat = np.empty((M, N))
        self.exp_z_zT_hat = np.empty((N, M, M))
        self.W_hat = np.random.randn(D, M)
        self.sigma_sq_hat: float = 1.0

    def E_step(self):
        bf_M = self.W_hat.T @ self.W_hat + self.sigma_sq_hat * np.eye(self.M)
        # TODO: Vectorize
        for n in range(self.N):
            self.exp_z_hat[:, n] = (
                np.linalg.inv(bf_M)
                @ self.W_hat.T
                @ (self.X[:, n].reshape(self.D, 1) - self.bar_x)
            ).reshape(self.M)
            exp_z_hat_n = self.exp_z_hat[:, n].reshape(self.M, 1)
            self.exp_z_zT_hat[n] = (
                self.sigma_sq_hat * np.linalg.inv(bf_M) + exp_z_hat_n @ exp_z_hat_n.T
            )

    def M_step(self):
        W_new = np.zeros_like(self.W_hat)
        left_addition = np.zeros((self.D, self.M))
        for n in range(self.N):
            E_z_n = self.exp_z_hat[:, n].reshape(self.M, 1)
            x_n = self.X[:, n].reshape(self.D, 1)
            left_addition += (x_n - self.bar_x) @ E_z_n.T
        W_new = left_addition @ np.linalg.inv(self.exp_z_zT_hat.sum(axis=0))
        assert W_new.shape == (self.D, self.M)
        sigma_sq_new = 0
        for n in range(self.N):
            x_n = self.X[:, n].reshape(self.D, 1)
            E_z_n = self.exp_z_hat[:, n].reshape(self.M, 1)
            addition = (
                (x_n - self.bar_x).T @ (x_n - self.bar_x)
                - 2 * E_z_n.T @ W_new.T @ (x_n - self.bar_x)
                + np.trace(self.exp_z_zT_hat[n] @ W_new.T @ W_new)
            )
            assert addition.shape == (1, 1), addition.shape
            sigma_sq_new += (addition[0, 0] / (self.N * self.D))
        self.W_hat, self.sigma_sq_hat = W_new, sigma_sq_new

    def EM(self, n_steps: int = 1, disable=False) -> None:
        for n in tqdm(list(range(n_steps)), leave=False, disable=disable):
            self.E_step()
            self.M_step()

    def statistics(self) -> pd.Series:
        return pd.Series(
            {
                "Distance from true W": np.linalg.norm(
                    self.W @ self.W.T - self.W_hat @ self.W_hat.T
                ),
                "Distance from true sigma_sq": np.abs(
                    self.sigma_sq - self.sigma_sq_hat
                ),
                "ELBO": self.ELBO(),
            }
        )

    def plot_EM(self, n_steps: int) -> Figure:
        df = []
        for n in tqdm(range(n_steps)):
            self.E_step()
            df.append(self.statistics().to_frame().T)
            self.M_step()
            df.append(self.statistics().to_frame().T)
            assert (
                not (np.isnan(self.W_hat).any()
                or np.isinf(self.W_hat).any()
                or np.isneginf(self.W_hat).any())
            ), self.W_hat
            assert (
                not (np.isnan(self.sigma_sq_hat)
                or np.isinf(self.sigma_sq_hat)
                or np.isneginf(self.sigma_sq_hat)
                or self.sigma_sq_hat == 0)
            ), self.sigma_sq_hat
            assert (
                not (np.isnan(self.exp_z_hat).any()
                or np.isinf(self.exp_z_hat).any()
                or np.isneginf(self.exp_z_hat).any())
            ), self.exp_z_hat
            assert (
                not (np.isnan(self.exp_z_zT_hat).any()
                or np.isinf(self.exp_z_zT_hat).any()
                or np.isneginf(self.exp_z_zT_hat).any())
            ), self.exp_z_zT_hat
        df = pd.concat(df, axis=0, ignore_index=True)
        fig = px.line(
            df,
            y=["Distance from true W", "Distance from true sigma_sq", "ELBO"],
            title="Distance from true parameters",
        )
        fig.show()
        return fig

    def ELBO(self) -> float:
        output = 0.0
        for n in range(self.N):
            output += -1 * (
                0.5 * self.D * np.log(2 * np.pi * self.sigma_sq_hat)
                + 0.5
                * self.M
                * np.log(
                    2 * np.pi
                )  # Note that this term is missing from Bishop, in Errata
                + 0.5 * np.trace(self.exp_z_zT_hat[n])
                + (0.5 / self.sigma_sq_hat)
                * (self.X[:, n].reshape(self.D, 1) - self.bar_x).T
                @ (self.X[:, n].reshape(self.D, 1) - self.bar_x)
                - (1 / self.sigma_sq_hat)
                * self.exp_z_hat[:, n].reshape(self.M, 1).T
                @ self.W_hat.T
                @ (self.X[:, n].reshape(self.D, 1) - self.bar_x)
                + (0.5 / self.sigma_sq_hat)
                * np.trace(self.exp_z_zT_hat[n] @ self.W_hat.T @ self.W_hat)
            )
            output += ( 
                0.5
                * np.log(
                    np.linalg.det(
                    self.exp_z_zT_hat[n] - self.exp_z_hat[:, n].reshape(self.M, 1) @ self.exp_z_hat[:, n].reshape(1, self.M)
                    )
                )
                + (self.M / 2) * (1 + np.log(2 * np.pi))
                
            )
        return float(output)
