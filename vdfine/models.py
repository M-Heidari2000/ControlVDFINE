import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import torch.nn.init as init
from torch.distributions import MultivariateNormal, Normal, Independent


class Encoder(nn.Module):
    """
        Probabilistic encoder: q(a_t|y_t) = N(mu(y_t), diag(sigma^2(y_t)))
    """

    def __init__(self, a_dim: int, y_dim: int, hidden_dim: int, min_var: float = 1e-3):
        super().__init__()

        self.a_dim = a_dim
        self.min_var = min_var

        self.backbone = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.mean_head = nn.Linear(hidden_dim, a_dim)
        self.logvar_head = nn.Linear(hidden_dim, a_dim)

    def forward(self, y: torch.Tensor) -> Independent:
        """
            Returns q(a_t|y_t) as an Independent Normal distribution.
        """
        h = self.backbone(y)
        mean = self.mean_head(h)
        var = F.softplus(self.logvar_head(h)) + self.min_var
        return Independent(Normal(mean, var.sqrt()), 1)


class Decoder(nn.Module):
    """
        a_t -> y_t
    """

    def __init__(self, a_dim: int, y_dim: int, hidden_dim: int):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(a_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, y_dim),
        )

    def forward(self, a: torch.Tensor):
        return self.mlp_layers(a)


class CostModel(nn.Module):
    def __init__(self, x_dim: int, u_dim: int):
        """
            Learnable quadratic cost function
        """
        super().__init__()

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.A = nn.Parameter(torch.eye(self.x_dim, dtype=torch.float32))
        self.q = nn.Parameter(torch.randn((1, self.x_dim), dtype=torch.float32))
        self.register_buffer("R", 1e-6 * torch.eye(self.u_dim, dtype=torch.float32))

    @property
    def Q(self):
        return self.A @ self.A.T

    def forward(self, x: torch.Tensor, u: torch.Tensor):
        res = x - self.q
        xQx = torch.einsum("bi,ij,bj->b", res, self.Q, res)
        uRu = torch.einsum("bi,ij,bj->b", u, self.R, u)
        cost = 0.5 * (xQx + uRu).reshape(-1, 1)
        return cost


class Dynamics(nn.Module):

    """
        Linear Gaussian state-space model with Kalman filtering.
        Supports analytical ELBO computation for the variational DFINE.
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        a_dim: int,
        hidden_dim: Optional[int] = 128,
        min_var: float = 1e-2,
        max_var: float = 1.0,
        locally_linear: Optional[bool] = False,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.a_dim = a_dim
        self._min_var = min_var
        self._max_var = max_var
        self.locally_linear = locally_linear

        if self.locally_linear:
            self.backbone = nn.Sequential(
                nn.Linear(x_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

            self.A_head = nn.Linear(hidden_dim, x_dim * x_dim)
            self.B_head = nn.Linear(hidden_dim, x_dim * u_dim)
            self.C_head = nn.Linear(hidden_dim, a_dim * x_dim)
            self.nx_head = nn.Linear(hidden_dim, x_dim)
            self.na_head = nn.Linear(hidden_dim, a_dim)
            self.alpha = nn.Parameter(torch.tensor([1e-2]))

            self._init_weights()
        else:
            self.A = nn.Parameter(torch.eye(x_dim))
            self.B = nn.Parameter(torch.randn(x_dim, u_dim))
            self.C = nn.Parameter(torch.randn(a_dim, x_dim))
            self.nx = nn.Parameter(torch.randn(x_dim))
            self.na = nn.Parameter(torch.randn(a_dim))

    def _init_weights(self):
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    init.zeros_(m.bias)

    def make_psd(self, P, eps=1e-6):
        b = P.shape[0]
        P = 0.5 * (P + P.transpose(-1, -2))
        P = P + eps * torch.eye(P.size(-1), device=P.device).expand(b, -1, -1)
        return P

    def get_dynamics(self, x):
        """
            get dynamics matrices depending on the state x
        """
        b = x.shape[0]

        if self.locally_linear:
            hidden = self.backbone(x)
            I = torch.eye(self.x_dim, device=x.device).expand(b, -1, -1)
            A = I + self.alpha * self.A_head(hidden).reshape(b, self.x_dim, self.x_dim)
            B = self.B_head(hidden).reshape(b, self.x_dim, self.u_dim)
            C = self.C_head(hidden).reshape(b, self.a_dim, self.x_dim)
            Nx = torch.diag_embed(self._min_var + (self._max_var - self._min_var) * torch.sigmoid(self.nx_head(hidden)))
            Na = torch.diag_embed(self._min_var + (self._max_var - self._min_var) * torch.sigmoid(self.na_head(hidden)))
        else:
            A = self.A.expand(b, -1, -1)
            B = self.B.expand(b, -1, -1)
            C = self.C.expand(b, -1, -1)
            Nx = torch.diag_embed(self._min_var + (self._max_var - self._min_var) * torch.sigmoid(self.nx)).expand(b, -1, -1)
            Na = torch.diag_embed(self._min_var + (self._max_var - self._min_var) * torch.sigmoid(self.na)).expand(b, -1, -1)

        return A, B, C, Nx, Na

    def get_a(self, x):
        """
        returns emissions (a) based on the input state (x)
        """

        _, _, C, _, _ = self.get_dynamics(x=x)
        return torch.einsum('bij,bj->bi', C, x)

    def prior(
        self,
        dist: MultivariateNormal,
        u: torch.Tensor,
    ):
        """
            single step dynamics update

            dist: N(b x, b x x)
            u: b u
        """

        mean = dist.loc
        cov = dist.covariance_matrix

        A, B, _, Nx, _ = self.get_dynamics(x=mean)

        next_mean = torch.einsum('bij,bj->bi', A, mean) + torch.einsum('bij,bj->bi', B, u)
        next_cov = torch.einsum('bij,bjk,bkl->bil', A, cov, A.transpose(1, 2)) + Nx
        next_cov = self.make_psd(next_cov)
        updated_dist = MultivariateNormal(loc=next_mean, covariance_matrix=next_cov)

        return updated_dist

    def posterior(
        self,
        dist: MultivariateNormal,
        a: torch.Tensor,
    ):
        """
            single step measurement update

            dist: N(b x, b x x)
            a: b a
        """

        mean = dist.loc
        cov = dist.covariance_matrix

        _, _, C, _, Na = self.get_dynamics(x=mean)

        S = torch.einsum('bij,bjk,bkl->bil', C, cov, C.transpose(1, 2)) + Na
        G = torch.einsum('bij,bjk,bkl->bil', cov, C.transpose(1, 2), torch.linalg.pinv(S))
        innovation = a - torch.einsum('bij,bj->bi', C, mean)
        next_mean = mean + torch.einsum('bij,bj->bi', G, innovation)
        next_cov = cov - torch.einsum('bij,bjk,bkl->bil', G, C, cov)
        next_cov = self.make_psd(next_cov)
        updated_dist = MultivariateNormal(loc=next_mean, covariance_matrix=next_cov)

        return updated_dist

    def compute_dynamics_kl(
        self,
        posterior_t: MultivariateNormal,
        posterior_t_minus_d: MultivariateNormal,
        u_seq: List[torch.Tensor],
        d: int,
    ) -> torch.Tensor:
        """
        Compute E_q(x_{t-d})[KL(q(x_t|.) || p(x_t|x_{t-d},.))].

        For locally_linear, falls back to KL(posterior_t || d_step_rollout).

        Args:
            posterior_t: q(x_t | a_{1:t}, u_{0:t-1})
            posterior_t_minus_d: q(x_{t-d} | a_{1:t-d}, u_{0:t-d-1})
            u_seq: [u_{t-d}, u_{t-d+1}, ..., u_{t-1}] (d elements)
            d: overshoot steps

        Returns:
            kl: (B,) expected KL per batch element
        """
        if self.locally_linear:
            rollout_dist = posterior_t_minus_d
            for j in range(d):
                rollout_dist = self.prior(dist=rollout_dist, u=u_seq[j])
            return torch.distributions.kl_divergence(posterior_t, rollout_dist)

        mu_x = posterior_t.loc
        Sigma_x = posterior_t.covariance_matrix
        mu_p = posterior_t_minus_d.loc
        Sigma_p = posterior_t_minus_d.covariance_matrix

        device = mu_x.device
        dx = self.x_dim

        A_mat = self.A
        B_mat = self.B
        Nx_diag = self._min_var + (self._max_var - self._min_var) * torch.sigmoid(self.nx)
        Nx = torch.diag(Nx_diag)

        # A^d
        A_d = torch.matrix_power(A_mat, d)

        # mu_bar_d = A^d mu_p + sum_{j=0}^{d-1} A^j B u_{t-1-j}
        # u_seq = [u_{t-d}, ..., u_{t-1}], so u_{t-1-j} = u_seq[d-1-j]
        mu_bar_d = mu_p @ A_d.T
        A_j = torch.eye(dx, device=device)
        for j in range(d):
            mu_bar_d = mu_bar_d + (u_seq[d - 1 - j] @ B_mat.T) @ A_j.T
            A_j = A_j @ A_mat

        # Sigma_d = sum_{j=0}^{d-1} A^j Nx (A^j)^T
        Sigma_d = torch.zeros(dx, dx, device=device)
        A_j = torch.eye(dx, device=device)
        for j in range(d):
            Sigma_d = Sigma_d + A_j @ Nx @ A_j.T
            A_j = A_j @ A_mat

        Sigma_d = Sigma_d + 1e-6 * torch.eye(dx, device=device)
        Sigma_d_inv = torch.linalg.inv(Sigma_d)

        # Mahalanobis: (mu_x - mu_bar_d)^T Sigma_d^{-1} (mu_x - mu_bar_d)
        diff = mu_x - mu_bar_d
        mahal = torch.einsum('bi,ij,bj->b', diff, Sigma_d_inv, diff)

        # tr(Sigma_d^{-1} A^d Sigma_p (A^d)^T)
        A_d_Sigma_p = torch.einsum('ij,bjk->bik', A_d, Sigma_p)
        A_d_Sigma_p_A_d_T = torch.einsum('bij,kj->bik', A_d_Sigma_p, A_d)
        trace_prior = torch.einsum('ij,bji->b', Sigma_d_inv, A_d_Sigma_p_A_d_T)

        # tr(Sigma_d^{-1} Sigma_x)
        trace_post = torch.einsum('ij,bji->b', Sigma_d_inv, Sigma_x)

        # log determinants
        log_det_Sigma_x = torch.logdet(Sigma_x)
        log_det_Sigma_d = torch.logdet(Sigma_d)

        kl = 0.5 * (mahal + trace_prior + trace_post - log_det_Sigma_x + log_det_Sigma_d - dx)

        return kl

    def compute_emission_nll(
        self,
        a_sample: torch.Tensor,
        mu_x: torch.Tensor,
        Sigma_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute -E_q(x_t|.)[log p(a_t|x_t)] where p(a_t|x_t) = N(C x_t, Na).

        Args:
            a_sample: (B, a_dim) sample from q(a_t|y_t)
            mu_x: (B, x_dim) posterior mean of x_t
            Sigma_x: (B, x_dim, x_dim) posterior covariance of x_t

        Returns:
            nll: (B,) negative expected log-likelihood
        """
        b = mu_x.shape[0]

        if self.locally_linear:
            _, _, C, _, Na = self.get_dynamics(x=mu_x)
        else:
            C = self.C.expand(b, -1, -1)
            Na_diag = self._min_var + (self._max_var - self._min_var) * torch.sigmoid(self.na)
            Na = torch.diag(Na_diag).expand(b, -1, -1)

        da = self.a_dim

        # C mu_x
        C_mu_x = torch.einsum('bij,bj->bi', C, mu_x)

        # Na^{-1}
        Na_inv = torch.linalg.inv(Na)

        # (a - C mu_x)^T Na^{-1} (a - C mu_x)
        diff = a_sample - C_mu_x
        mahal = torch.einsum('bi,bij,bj->b', diff, Na_inv, diff)

        # tr(C^T Na^{-1} C Sigma_x)
        Ct_Na_inv_C = torch.einsum('bji,bjk,bkl->bil', C, Na_inv, C)
        trace_term = torch.einsum('bij,bji->b', Ct_Na_inv_C, Sigma_x)

        # log det Na
        log_det_Na = torch.logdet(Na)

        nll = 0.5 * (da * torch.log(torch.tensor(2 * torch.pi, device=mu_x.device)) + log_det_Na + mahal + trace_term)

        return nll

    def predict_future(
        self,
        x_samples: torch.Tensor,
        u_future: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Analytically compute future emission distributions for the predictive ELBO.
        Given fixed samples x_hat_t, compute p(a_{t+k} | x_hat_t, u) for each future step k.

        Args:
            x_samples: (M, B, x_dim) samples from context posterior
            u_future: (K, B, u_dim) future control inputs

        Returns:
            a_means: list of K tensors, each (M, B, a_dim)
            a_covs: list of K tensors, each (a_dim, a_dim)
        """
        if self.locally_linear:
            raise NotImplementedError("Analytical predictive ELBO not supported for locally_linear")

        K = u_future.shape[0]
        device = x_samples.device

        A_mat = self.A
        B_mat = self.B
        C_mat = self.C
        Nx_diag = self._min_var + (self._max_var - self._min_var) * torch.sigmoid(self.nx)
        Nx = torch.diag(Nx_diag)
        Na_diag = self._min_var + (self._max_var - self._min_var) * torch.sigmoid(self.na)
        Na = torch.diag(Na_diag)

        a_means = []
        a_covs = []

        # Accumulated control input: s_h = A s_{h-1} + B u_{future[h-1]}
        # x_samples shape: (M, B, x_dim)
        M, B_size, _ = x_samples.shape
        ctrl_acc = torch.zeros(B_size, self.x_dim, device=device)

        # A^h accumulated
        A_pow = torch.eye(self.x_dim, device=device)

        # Process noise accumulated: Sigma_h = sum_{j=0}^{h-1} A^j Nx (A^j)^T
        Sigma_acc = torch.zeros(self.x_dim, self.x_dim, device=device)
        A_pow_sigma = torch.eye(self.x_dim, device=device)

        for h in range(1, K + 1):
            # Control recurrence: s_h = A @ s_{h-1} + B @ u_future[h-1]
            ctrl_acc = ctrl_acc @ A_mat.T + u_future[h - 1] @ B_mat.T

            # A^h
            A_pow = A_pow @ A_mat

            # State mean: mu_{t+h} = A^h @ x_samples + s_h
            state_mean = torch.einsum('ij,mbj->mbi', A_pow, x_samples) + ctrl_acc.unsqueeze(0)

            # Process noise: add j=h-1 term
            Sigma_acc = Sigma_acc + A_pow_sigma @ Nx @ A_pow_sigma.T
            A_pow_sigma = A_pow_sigma @ A_mat

            # Emission: a_mean = C @ state_mean, a_cov = C @ Sigma_acc @ C^T + Na
            a_mean = torch.einsum('ij,mbj->mbi', C_mat, state_mean)
            a_cov = C_mat @ Sigma_acc @ C_mat.T + Na

            a_means.append(a_mean)
            a_covs.append(a_cov)

        return a_means, a_covs

    def generate(self, dist: MultivariateNormal, u: torch.Tensor):
        """
            generates trajectory given the initial belief and list of actions
            uses mean decoding
        """

        with torch.no_grad():
            samples = []

            if u.dim() == 2:
                u = u.unsqueeze(0)
            d, _, _ = u.shape

            for l in range(d):
                dist = self.prior(dist=dist, u=u[l])
                samples.append(dist.loc)

            samples = torch.stack(samples, dim=0)

        return samples

    def forward(self, u: torch.Tensor, a: torch.Tensor):
        """
            multi step inference of priors and posteriors

            inputs:
                - a: a0:T-1
                - u: u0:T-1
            outputs:
                priors: one step priors over the states
                posteriors: posterior over the states

            Notes: u[T-1] is not used
        """

        T, B, _ = u.shape
        device = u.device
        prior = MultivariateNormal(
            loc=torch.zeros((B, self.x_dim), device=device),
            covariance_matrix=torch.eye(self.x_dim, device=device).expand(B, -1, -1),
        )
        posterior = self.posterior(dist=prior, a=a[0])

        priors = [prior]
        posteriors = [posterior]

        for t in range(T-1):
            prior = self.prior(dist=posterior, u=u[t])
            posterior = self.posterior(dist=prior, a=a[t+1])
            priors.append(prior)
            posteriors.append(posterior)

        return priors, posteriors
