"""
flu.applications.bo.gp
========================
Lightweight Gaussian Process with RBF kernel.
Pure numpy / scipy — no sklearn or gpytorch dependency.

Implements:
  GaussianProcess — fit, predict (mean + std), log_marginal_likelihood
  expected_improvement(mu, sigma, f_best) — EI acquisition function
  optimize_hyperparams(gp, X, y) — MLE via scipy.optimize.minimize

Design choice: keep it simple and transparent so the BO loop is easy
to read and the performance differences come from the sampler, not
from black-box GP magic.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.special import ndtr   # standard normal CDF


# ── Kernel ─────────────────────────────────────────────────────────────────────

def _rbf(X1: np.ndarray, X2: np.ndarray,
         length_scale: float, sigma_f: float) -> np.ndarray:
    """Squared-exponential (RBF) kernel: k(x,x') = σ_f² exp(-‖x-x'‖²/(2ℓ²))."""
    diff = X1[:, None, :] - X2[None, :, :]
    sq   = np.sum(diff**2, axis=-1)
    return sigma_f**2 * np.exp(-0.5 * sq / (length_scale**2))


# ── GP class ───────────────────────────────────────────────────────────────────

class GaussianProcess:
    """
    Minimal GP regression with RBF kernel and noise variance.

    Parameters
    ----------
    length_scale : float   RBF length scale (same for all dims, isotropic)
    sigma_f      : float   Signal std (kernel amplitude)
    sigma_n      : float   Observation noise std
    """

    def __init__(self,
                 length_scale: float = 0.3,
                 sigma_f: float = 1.0,
                 sigma_n: float = 1e-3):
        self.length_scale = length_scale
        self.sigma_f      = sigma_f
        self.sigma_n      = sigma_n
        self._X_train: np.ndarray | None = None
        self._alpha:   np.ndarray | None = None
        self._L:       np.ndarray | None = None

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        """Fit GP to training data (X, y). Overwrites previous training set."""
        self._X_train = X.copy()
        K = _rbf(X, X, self.length_scale, self.sigma_f)
        K += self.sigma_n**2 * np.eye(len(X))
        try:
            c, low  = cho_factor(K)
            self._L = c
            self._low = low
            self._alpha = cho_solve((c, low), y)
        except np.linalg.LinAlgError:
            # Fallback: add more jitter
            K += 1e-6 * np.eye(len(X))
            c, low  = cho_factor(K)
            self._L = c
            self._low = low
            self._alpha = cho_solve((c, low), y)
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, X_star: np.ndarray,
                return_std: bool = True) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Predict mean (and std) at X_star."""
        assert self._X_train is not None, "Call fit() first"
        K_s  = _rbf(self._X_train, X_star, self.length_scale, self.sigma_f)
        mu   = K_s.T @ self._alpha

        if not return_std:
            return mu

        K_ss = _rbf(X_star, X_star, self.length_scale, self.sigma_f)
        v    = cho_solve((self._L, self._low), K_s)
        var  = np.diag(K_ss) - np.sum(K_s * v, axis=0)
        var  = np.maximum(var, 1e-12)   # numerical safety
        return mu, np.sqrt(var)

    # ── log marginal likelihood ────────────────────────────────────────────────

    def log_marginal_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute log p(y | X, θ) for hyperparameter tuning."""
        K = _rbf(X, X, self.length_scale, self.sigma_f)
        K += self.sigma_n**2 * np.eye(len(X))
        try:
            c, low = cho_factor(K)
        except np.linalg.LinAlgError:
            return -1e10
        alpha = cho_solve((c, low), y)
        # Log|K| = 2 * sum(log diag(L))
        log_det = 2 * np.sum(np.log(np.diag(c if not low else c.T)))
        n       = len(y)
        return float(-0.5 * (y @ alpha + log_det + n * np.log(2 * np.pi)))


# ── Hyperparameter optimisation ─────────────────────────────────────────────────

def optimize_hyperparams(X: np.ndarray, y: np.ndarray,
                         n_restarts: int = 3) -> GaussianProcess:
    """
    Maximise log marginal likelihood over (log ℓ, log σ_f, log σ_n).
    Returns a freshly fitted GP with optimised hyperparameters.
    """
    D = X.shape[1]
    best_lml = -np.inf
    best_gp  = None

    rng = np.random.default_rng(0)
    # Initial guesses: random in log-space + one sensible default
    starts = [(0.0, 0.0, -3.0)]   # log(l)=1, log(σ_f)=1, log(σ_n)=0.05
    for _ in range(n_restarts - 1):
        starts.append(rng.uniform(-2, 1, 3).tolist())

    for x0 in starts:
        def neg_lml(params):
            l, sf, sn = np.exp(params)
            gp = GaussianProcess(length_scale=l, sigma_f=sf, sigma_n=max(sn, 1e-6))
            return -gp.log_marginal_likelihood(X, y)

        res = minimize(neg_lml, x0, method="L-BFGS-B",
                       bounds=[(-4, 2), (-3, 3), (-6, 0)],
                       options={"maxiter": 100, "ftol": 1e-6})
        l, sf, sn = np.exp(res.x)
        gp = GaussianProcess(length_scale=l, sigma_f=sf, sigma_n=max(sn, 1e-6))
        gp.fit(X, y)
        lml = gp.log_marginal_likelihood(X, y)
        if lml > best_lml:
            best_lml = lml
            best_gp  = gp

    return best_gp


# ── Acquisition function ─────────────────────────────────────────────────────

def expected_improvement(mu: np.ndarray,
                          sigma: np.ndarray,
                          f_best: float,
                          xi: float = 0.01) -> np.ndarray:
    """
    Expected Improvement (EI) for minimisation.

        EI(x) = (f_best - mu - ξ) · Φ(Z) + σ · φ(Z)
        Z = (f_best - mu - ξ) / σ

    Returns EI values (higher = more promising candidate).
    """
    imp   = f_best - mu - xi
    Z     = imp / (sigma + 1e-9)
    phi_Z = np.exp(-0.5 * Z**2) / np.sqrt(2 * np.pi)   # standard normal pdf
    Phi_Z = ndtr(Z)                                       # standard normal CDF
    ei    = imp * Phi_Z + sigma * phi_Z
    ei[sigma < 1e-9] = 0.0
    return ei
