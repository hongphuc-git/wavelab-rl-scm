"""Native NumPy/SciPy implementation of the Chebyshev Spectral Collocation Method (SCM).

Translates chebdif.m, voigt2tensor.m, build_cylinder_matrices.m (Model 1, L-mode, m=0),
and export_scm_for_fem.m / polyeig directly to NumPy + SciPy LAPACK so training and
evaluation run identically on Google Colab (Linux/GPU/CPU) without requiring a MATLAB license.
"""
import numpy as np
from scipy.linalg import eig, toeplitz


def chebdif(N: int, M: int = 2):
    """Chebyshev-Gauss-Lobatto nodes and differentiation matrices (Weideman & Reddy 2000)."""
    I = np.eye(N, dtype=float)
    L = I.astype(bool)
    n1 = N // 2
    n2 = (N + 1) // 2
    k = np.arange(N, dtype=float).reshape(-1, 1)
    th = k * np.pi / (N - 1)
    x = np.sin(np.pi * np.arange(N - 1, -N, -2, dtype=float) / (2 * (N - 1)))

    T = np.tile(th / 2.0, (1, N))
    DX = 2.0 * np.sin(T.T + T) * np.sin(T.T - T)
    DX = np.vstack([DX[:n1, :], -np.flipud(np.fliplr(DX[:n2, :]))])
    DX[L] = 1.0

    C = toeplitz((-1.0) ** k.ravel()).astype(float)
    C[0, :] *= 2.0
    C[-1, :] *= 2.0
    C[:, 0] /= 2.0
    C[:, -1] /= 2.0

    Z = 1.0 / DX
    Z[L] = 0.0

    D = np.eye(N, dtype=float)
    DM = np.zeros((N, N, M), dtype=float)
    for ell in range(1, M + 1):
        D = ell * Z * (C * np.tile(np.diag(D).reshape(-1, 1), (1, N)) - D)
        D[L] = -np.sum(D, axis=1)
        DM[:, :, ell - 1] = D
    return x, DM


def voigt2tensor(C: np.ndarray) -> np.ndarray:
    """Convert 6x6 Voigt stiffness matrix to 3x3x3x3 tensor c_{ijkl}."""
    V = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]], dtype=int)
    C_mat = np.zeros((3, 3, 3, 3), dtype=C.dtype)
    for i in range(3):
        for j in range(3):
            vi = V[i, j]
            for k in range(3):
                for l in range(3):
                    C_mat[i, j, k, l] = C[vi, V[k, l]]
    return C_mat


def iso_tensor_Enu(E: float, nu: float) -> np.ndarray:
    """3x3x3x3 isotropic stiffness tensor from E and nu."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C = np.array([
        [lam + 2 * mu, lam, lam, 0.0, 0.0, 0.0],
        [lam, lam + 2 * mu, lam, 0.0, 0.0, 0.0],
        [lam, lam, lam + 2 * mu, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, mu, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, mu, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, mu],
    ], dtype=float)
    return voigt2tensor(C)


def build_cylinder_matrices_L0(E: float, rho: float, nu: float, N: int, a: float, b: float):
    """Assemble SCM matrices (L2, L1, L0, M) for Model 1 (free hollow pipe), m=0, mode='L'."""
    h = b - a
    C = iso_tensor_Enu(E, nu)
    n_dof = 3
    G = np.zeros((3, 3), dtype=float)
    G[0, 1] = -1.0
    G[1, 0] = 1.0
    I3 = np.eye(3, dtype=float)
    Gm = G  # m = 0

    y, Dy = chebdif(N, 2)
    D1 = (2.0 / h) * Dy[:, :, 0]
    D2 = (2.0 / h) ** 2 * Dy[:, :, 1]
    r = (h * y + b + a) / 2.0
    Rinv = np.diag(1.0 / r)
    R2inv = np.diag(1.0 / (r ** 2))
    Id = np.eye(N, dtype=float)

    C11 = C[0, :, :, 0]
    C12 = C[0, :, :, 1]
    C13 = C[0, :, :, 2]
    C21 = C[1, :, :, 0]
    C22 = C[1, :, :, 1]
    C23 = C[1, :, :, 2]
    C31 = C[2, :, :, 0]
    C32 = C[2, :, :, 1]
    C33 = C[2, :, :, 2]

    Ms2 = np.kron(C33, Id)
    Ms1 = np.kron(C13 + C31, D1) + np.kron(C13 + Gm @ C23 + C32 @ Gm, Rinv)
    Ms0 = (np.kron(C11, D2)
           + np.kron(C11 + C12 @ Gm + Gm @ C21, Rinv @ D1)
           + np.kron(Gm @ C22 @ Gm, R2inv))

    bc_idx = [0, N - 1]
    B1 = np.kron(C13, Id[bc_idx, :])
    B0 = np.kron(C11, D1[bc_idx, :]) + np.kron(C12 @ Gm, Rinv[bc_idx, :])
    Mm = np.kron(rho * I3, Id)

    bc_rows = []
    for d in range(n_dof):
        bc_rows.extend([d * N, (d + 1) * N - 1])
    Ms2[bc_rows, :] = 0.0
    Ms1[bc_rows, :] = B1
    Ms0[bc_rows, :] = B0
    Mm[bc_rows, :] = 0.0

    # Longitudinal mode reduction: remove u_theta (DOF 2 -> indices N..2N-1), keep u_r and u_z
    keep = np.r_[0:N, 2 * N:3 * N]
    Ms2 = Ms2[np.ix_(keep, keep)]
    Ms1 = Ms1[np.ix_(keep, keep)]
    Ms0 = Ms0[np.ix_(keep, keep)]
    Mm = Mm[np.ix_(keep, keep)]

    norm_s = np.max(np.abs(Ms2))
    if norm_s == 0.0:
        norm_s = np.max(np.abs(Ms0)) or 1.0
    return Ms2 / norm_s, Ms1 / norm_s, Ms0 / norm_s, Mm / norm_s


def solve_scm_dispersion(m, nodes: int = 30, frequencies: int = 601, curve_points: int = 601,
                         b_outer: float = 0.008, rho: float = 1000.0):
    """Compute dispersion roots/curves in pure NumPy/SciPy matching rl_worker.m."""
    N = int(nodes)
    h, c0, nu = float(m[0]), float(m[1]), float(m[2])
    a = b_outer - h / 1000.0
    E = rho * (1000.0 * c0) ** 2
    L2, L1, L0, M = build_cylinder_matrices_L0(E, rho, nu, N, a, b_outer)

    if curve_points and curve_points > 0:
        n_k = int(curve_points)
        k_vec = np.linspace(0.01, 4000.0, n_k)
        # Static condensation (Schur complement) on the 4 traction-free boundary rows
        # where M[bc, :] == 0 and L2[bc, :] == 0, enabling batched LAPACK eigvals.
        bc = np.array([0, N - 1, N, 2 * N - 1], dtype=int)
        inter = np.setdiff1d(np.arange(2 * N, dtype=int), bc)
        m_interior = float(M[inter[0], inter[0]])
        ik = 1j * k_vec[:, None, None]
        A_all = (ik ** 2) * L2 + ik * L1 + L0
        A_BB = A_all[:, bc[:, None], bc[None, :]]
        A_BI = A_all[:, bc[:, None], inter[None, :]]
        A_IB = A_all[:, inter[:, None], bc[None, :]]
        A_II = A_all[:, inter[:, None], inter[None, :]]
        S_all = (A_II - A_IB @ np.linalg.solve(A_BB, A_BI)) / (-m_interior)
        w2_all = np.linalg.eigvals(S_all)

        freqs_by_k = []
        max_modes = 0
        eps_sqrt = np.sqrt(np.finfo(float).eps)
        for w2 in w2_all:
            finite = np.isfinite(w2)
            if not finite.any():
                freqs_by_k.append(np.empty(0, dtype=float))
                continue
            w2_fin = w2[finite]
            scale = max(float(np.max(np.abs(w2_fin))), 1.0)
            good = (np.real(w2_fin) > 0.0) & (np.abs(np.imag(w2_fin)) <= 1e-8 * scale)
            w2r = np.real(w2_fin[good])
            if w2r.size == 0:
                freqs_by_k.append(np.empty(0, dtype=float))
                continue
            wn = np.sqrt(w2r)
            rb = eps_sqrt * max(1.0, float(np.max(wn)))
            wn = wn[wn >= rb]
            f = wn / (2.0 * np.pi)
            f = np.sort(f[(f >= 1000.0) & (f <= 2e6)])
            freqs_by_k.append(f)
            if f.size > max_modes:
                max_modes = f.size
        if max_modes == 0:
            raise RuntimeError('SCM returned no dense dispersion curves')
        f_all = np.full((max_modes, n_k), np.nan, dtype=float)
        for ii, fi in enumerate(freqs_by_k):
            if fi.size:
                f_all[:fi.size, ii] = fi
        valid = np.isfinite(f_all)
        k_grid = np.broadcast_to(k_vec.reshape(1, -1), f_all.shape)
        ff = f_all.T[valid.T]
        kk = k_grid.T[valid.T]
        bins = np.round((ff - 1000.0) / (2e6 - 1000.0) * max(int(frequencies) - 1, 1)) + 1.0
        return dict(ff=ff, kk=kk, bins=bins, curve_k=k_vec, curve_f=f_all)

    # Fixed-frequency quadratic eigenvalue mode: polyeig(L0 + w^2 M, L1, L2)
    f_eval = np.linspace(1e3, 2e6, int(frequencies))
    n = L0.shape[0]
    Z = np.zeros((n, n), dtype=float)
    I_n = np.eye(n, dtype=float)
    B_comp = np.block([[I_n, Z], [Z, -L2]])
    ff_list, kk_list, bins_list = [], [], []
    for j, fj in enumerate(f_eval, start=1):
        w = 2.0 * np.pi * fj
        A_comp = np.block([[Z, I_n], [L0 + (w ** 2) * M, L1]])
        z = eig(A_comp, B_comp, right=False, check_finite=False)
        k = -1j * z
        good = (np.isfinite(k)
                & (np.real(k) >= 0.0)
                & (np.real(k) <= 4000.0)
                & (np.abs(np.imag(k)) <= 1e-6 * np.maximum(1.0, np.abs(np.real(k)))))
        k_good = np.sort(np.real(k[good]))
        if k_good.size:
            kk_list.append(k_good)
            ff_list.append(np.full(k_good.size, fj))
            bins_list.append(np.full(k_good.size, float(j)))
    if not kk_list:
        raise RuntimeError('SCM returned no propagating roots')
    return dict(ff=np.concatenate(ff_list), kk=np.concatenate(kk_list), bins=np.concatenate(bins_list))
