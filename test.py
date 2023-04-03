from ppca import PPCA
import numpy as np
from numpy.linalg import inv
from tqdm.auto import tqdm

def test_E_step():
    p = PPCA(2, 2, 1_000, np.eye(2), 0.0)
    p.W_hat = np.eye(2)
    p.sigma_sq_hat = 0.0
    p.E_step()
    assert np.allclose(p.exp_z_hat.mean(axis=1), np.zeros((1, 2)))

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

def test_ELBO_finds_best_max():
    np.random.seed(0)
    p = PPCA(2, 2, 10, np.eye(2), 1.0)
    p.W_hat = np.eye(2)
    p.sigma_sq_hat = 1.0
    p.E_step()
    best_likelihood = p.ELBO()
    for _ in tqdm(range(100)):
        p.W_hat = np.random.randn(2, 2)
        p.sigma_sq_hat = np.random.rand()
        assert p.ELBO() <= best_likelihood, f"{p.ELBO()} > {best_likelihood}"

def test_ELBO_nondecreasing_1():
    np.random.seed(0)
    p = PPCA(20, 4, 10, np.random.randn(20, 4), 3.0)
    last_elbo = -np.inf
    for i in tqdm(range(100)):
        p.E_step()
        assert p.ELBO() >= last_elbo , f"{p.ELBO()} <= {last_elbo} on iteration {i}"  # the 1% increase is to avoid numerical issues
        last_elbo = p.ELBO()
        p.M_step()
        assert p.ELBO() >= last_elbo, f"{p.ELBO()} <= {last_elbo} on iteration {i}"  # the 1% increase is to avoid numerical issues

def test_ELBO_nondecreasing_2():
    np.random.seed(0)
    p = PPCA(2, 1, 3, np.random.randn(2, 1), 3.0)
    last_elbo = -np.inf
    for i in tqdm(range(100)):
        p.E_step()
        assert p.ELBO() >= last_elbo , f"{p.ELBO()} <= {last_elbo} on iteration {i}"  # the 1% increase is to avoid numerical issues
        last_elbo = p.ELBO()
        p.M_step()
        assert p.ELBO() >= last_elbo, f"{p.ELBO()} <= {last_elbo} on iteration {i}"  # the 1% increase is to avoid numerical issues






