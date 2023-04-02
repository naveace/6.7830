#%%
from typing import Dict
from numpy.linalg import inv
from tqdm.auto import tqdm
import numpy as np
import plotly.express as px
import pandas as pd

#%%
class PPCA:
    def __init__(self, D: int, M: int, N: int, W: np.ndarray, sigma_sq: float) -> None:
        self.D = D
        self.M = M
        self.N = N
        self.W = W  # N X M
        self.sigma_sq = sigma_sq
        self.Z = np.random.randn(M, N)
        self.X: np.ndarray = self.W @ self.Z + np.random.randn(D, N) * np.sqrt(
            self.sigma_sq
        )
        self.bar_x = self.X.mean(axis=1).reshape(D, 1)
        self.true_exp_z = (
            inv(self.W.T @ self.W + self.sigma_sq * np.eye(M))
            @ self.W.T
            @ (self.X - self.bar_x)
        )
        self.true_exp_z_zT = np.empty((N, M, M))
        for n in range(N):
            self.true_exp_z_zT[n] = (
                self.sigma_sq * inv(self.W.T @ self.W + self.sigma_sq * np.eye(M))
                + self.true_exp_z[:, n].reshape(1, M)
                @ self.true_exp_z[:, n].reshape(1, M).T
            )
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
            sigma_sq_new += addition[0, 0] / (self.N * self.D)
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

    def plot_EM(self, n_steps: int) -> None:
        df = []
        for n in tqdm(range(n_steps)):
            self.E_step()
            df.append(self.statistics().to_frame().T)
            self.M_step()
            df.append(self.statistics().to_frame().T)
            assert not np.isnan(self.W_hat).any() or np.isinf(self.W_hat).any() or np.isneginf(self.W_hat).any(), self.W_hat
            assert not np.isnan(self.sigma_sq_hat) or np.isinf(self.sigma_sq_hat) or np.isneginf(self.sigma_sq_hat), self.sigma_sq_hat
            assert not np.isnan(self.exp_z_hat).any() or np.isinf(self.exp_z_hat).any() or np.isneginf(self.exp_z_hat).any(), self.exp_z_hat
            assert not np.isnan(self.exp_z_zT_hat).any() or np.isinf(self.exp_z_zT_hat).any() or np.isneginf(self.exp_z_zT_hat).any(), self.exp_z_zT_hat
        df = pd.concat(df, axis=0, ignore_index=True)
        fig = px.line(
            df,
            y=["Distance from true W", "Distance from true sigma_sq", "ELBO"],
            title="Distance from true parameters",
        )
        fig.show()

    def ELBO(self) -> float:
        output = 0.0
        for n in range(self.N):
            output += (
                -0.5 * self.D * np.log(2 * np.pi * self.sigma_sq_hat)
                - 0.5 * self.M * np.log(2 * np.pi)
                - 0.5 * np.trace(self.exp_z_zT_hat[n])
                - (0.5 / self.sigma_sq_hat)
                * (self.X[:, n].reshape(self.D, 1) - self.bar_x).T
                @ (self.X[:, n].reshape(self.D, 1) - self.bar_x)
                + (1 / self.sigma_sq_hat)
                * self.exp_z_hat[:, n].reshape(self.M, 1).T
                @ self.W_hat.T
                @ (self.X[:, n].reshape(self.D, 1) - self.bar_x)
                - (0.5 / self.sigma_sq_hat)
                * np.trace(self.exp_z_zT_hat[n] @ self.W_hat.T @ self.W_hat)
            )
        # Now we add the entropy of the Z distribution
        output += sum(
            0.5 * self.M * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(self.exp_z_zT_hat[n])) + 0.5 * self.M
            for n in range(self.N)
        )
        return float(output)


#%%
def test_E_step():
    p = PPCA(2, 2, 1_000, np.eye(2), 0.0)
    p.W_hat = np.eye(2)
    p.sigma_sq_hat = 0.0
    p.E_step()
    assert np.allclose(p.exp_z_hat.mean(axis=1), np.zeros((1, 2)))


test_E_step()
#%%
def test_M_step():
    p = PPCA(2, 2, 1_000, np.eye(2), 0.0)
    p.exp_z_hat = inv(p.W.T @ p.W + p.sigma_sq * np.eye(2)) @ p.W.T @ (p.X - p.bar_x)
    p.exp_z_zT_hat = np.concatenate(
        [
            (
                p.sigma_sq * inv(p.W.T @ p.W + p.sigma_sq * np.eye(2))
                + p.exp_z_hat[:, n].reshape(p.M, 1)
                @ p.exp_z_hat[:, n].reshape(p.M, 1).T
            ).reshape(1, p.M, p.M)
            for n in range(p.exp_z_hat.shape[1])
        ],
        axis=0,
    )
    p.M_step()
    assert np.allclose(p.W_hat, np.eye(2))
    assert np.allclose(p.sigma_sq_hat, 0.0)


test_M_step()
#%%
def test_ELBO():
    np.random.seed(0)
    p = PPCA(2, 2, 1_000, np.eye(2), 1.0)
    p.W_hat = np.eye(2)
    p.sigma_sq_hat = 1.0
    p.E_step()
    best_likelihood = p.ELBO()
    for _ in tqdm(range(100)):
        p.W_hat = np.random.randn(2, 2)
        p.sigma_sq_hat = np.random.rand()
        assert (
            p.ELBO() <= best_likelihood
        ), f"{p.ELBO()} > {best_likelihood}"
    p = PPCA(20, 4, 10_000, np.random.randn(20, 4), 3.0)
    last_elbo = -np.inf
    for i in tqdm(range(100)):
        p.E_step()
        assert p.ELBO() > last_elbo * (1.01) , f"{p.ELBO()} <= {last_elbo} on iteration {i}" # the 1% increase is to avoid numerical issues
        last_elbo = p.ELBO()
        p.M_step()


test_ELBO()
# %%
p = PPCA(2, 1, 2, np.array([1, 0]).reshape(2, 1), 0.00000001)
p.plot_EM(100)

# %%
p = PPCA(2, 1, 100_000, np.array([1, 1]).reshape(2, 1), 1.0)
p.plot_EM(30)
# %%
px.scatter(
    pd.DataFrame(p.X.T),
    x=0,
    y=1,
).show()
# %%
p = PPCA(3, 1, 10_000, np.array([1, 1, 1]).reshape(3, 1), 1.0)
p.plot_EM(30)
# %%
