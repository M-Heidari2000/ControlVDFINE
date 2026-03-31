import torch
import wandb
import einops
import torch.nn as nn
from tqdm import tqdm
from omegaconf.dictconfig import DictConfig
from .memory import ReplayBuffer
from .utils import bottle_mvn
from torch.nn.utils import clip_grad_norm_
from .models import (
    Encoder,
    Decoder,
    Dynamics,
    CostModel,
)


def compute_elbo_losses(
    encoder: Encoder,
    decoder: Decoder,
    dynamics_model: Dynamics,
    y: torch.Tensor,
    u: torch.Tensor,
    config: DictConfig,
):
    """
    Compute all variational ELBO loss terms.

    Args:
        y: (chunk_length, B, y_dim) observations
        u: (chunk_length, B, u_dim) controls
        config: training config

    Returns:
        dict of loss terms (all scalars, to be minimized)
    """
    T = y.shape[0]
    B = y.shape[1]
    device = y.device

    context_len = T - config.prediction_k
    overshoot_d = config.overshoot_d

    # ---- Context window: encode and filter ----
    y_context = y[:context_len]
    u_context = u[:context_len]

    # Probabilistic encoding: q(a_t|y_t)
    y_flat = einops.rearrange(y_context, 'l b y -> (l b) y')
    encoder_dists = encoder(y_flat)
    a_samples = encoder_dists.rsample()
    a_samples_seq = einops.rearrange(a_samples, '(l b) a -> l b a', b=B)

    # Encoder entropy: H(q(a_t|y_t))
    encoder_entropy = encoder_dists.entropy()
    encoder_entropy = einops.rearrange(encoder_entropy, '(l b) -> l b', b=B)

    # Kalman filtering on context
    priors, posteriors = dynamics_model(a=a_samples_seq, u=u_context)

    # ---- Term 1: Reconstruction loss E_q(a|y)[log p(y|a)] ----
    # Using MSE (proportional to -log p under unit-variance Gaussian decoder)
    y_recon = decoder(a_samples)
    recon_loss = nn.MSELoss()(y_recon, y_flat)

    # ---- Term 2: Dynamics KL ----
    # E_q(x_{t-d})[KL(q(x_t|.) || p(x_t|x_{t-d},.))]
    dynamics_kl = torch.tensor(0.0, device=device)
    dynamics_kl_count = 0
    for t in range(overshoot_d, context_len):
        u_seq = [u_context[t - overshoot_d + j] for j in range(overshoot_d)]
        kl_t = dynamics_model.compute_dynamics_kl(
            posterior_t=posteriors[t],
            posterior_t_minus_d=posteriors[t - overshoot_d],
            u_seq=u_seq,
            d=overshoot_d,
        )
        dynamics_kl = dynamics_kl + kl_t.clamp(min=config.kl_free_nats).mean()
        dynamics_kl_count += 1
    if dynamics_kl_count > 0:
        dynamics_kl = dynamics_kl / dynamics_kl_count

    # ---- Term 3: Emission loss ----
    # -E_q[log q(a_t|y_t)/p(a_t|x_t)] = H(q(a|y)) + E_q[log p(a_t|x_t)]
    # Note: this is NOT a KL divergence because q(x_t|a_{1:t},u) depends on a_t,
    # so the expectations over q(a|y) and q(x|a,u) cannot be swapped.
    # Loss contribution (neg ELBO): emission_nll - entropy
    emission_loss = torch.tensor(0.0, device=device)
    for t in range(context_len):
        nll_t = dynamics_model.compute_emission_nll(
            a_sample=a_samples_seq[t],
            mu_x=posteriors[t].loc,
            Sigma_x=posteriors[t].covariance_matrix,
        )
        emission_loss = emission_loss + (nll_t - encoder_entropy[t]).mean()
    emission_loss = emission_loss / context_len

    # ---- Term 4: Predictive ELBO ----
    pred_loss = torch.tensor(0.0, device=device)
    if config.prediction_k > 0:
        last_posterior = posteriors[context_len - 1]
        M = config.num_pred_samples
        x_samples = last_posterior.rsample((M,))  # (M, B, x_dim)

        u_future = u[context_len - 1:context_len - 1 + config.prediction_k]
        y_future = y[context_len:context_len + config.prediction_k]

        a_means, a_covs = dynamics_model.predict_future(x_samples, u_future)

        for k in range(config.prediction_k):
            # Reparameterized sample from predicted emission
            L = torch.linalg.cholesky(
                a_covs[k] + 1e-6 * torch.eye(dynamics_model.a_dim, device=device)
            )
            eps = torch.randn_like(a_means[k])
            a_pred = a_means[k] + eps @ L.T  # (M, B, a_dim)

            # Decode and compute reconstruction
            y_pred = decoder(einops.rearrange(a_pred, 'm b a -> (m b) a'))
            y_pred = einops.rearrange(y_pred, '(m b) y -> m b y', m=M)

            y_target = y_future[k].unsqueeze(0).expand(M, -1, -1)
            pred_loss = pred_loss + nn.MSELoss()(y_pred, y_target)

        pred_loss = pred_loss / config.prediction_k

    # ---- Total loss (negative ELBO) ----
    total_loss = (
        recon_loss
        + config.kl_dynamics_weight * dynamics_kl
        + config.kl_emission_weight * emission_loss
        + config.pred_weight * pred_loss
    )

    return {
        "recon_loss": recon_loss,
        "dynamics_kl": dynamics_kl,
        "emission_loss": emission_loss,
        "pred_loss": pred_loss,
        "total_loss": total_loss,
    }


def train_backbone(
    config: DictConfig,
    train_buffer: ReplayBuffer,
    test_buffer: ReplayBuffer,
):

    # define models and optimizer
    device = "cuda" if (torch.cuda.is_available() and not config.disable_gpu) else "cpu"

    encoder = Encoder(
        y_dim=train_buffer.y_dim,
        a_dim=config.a_dim,
        hidden_dim=config.hidden_dim,
        min_var=config.encoder_min_var,
    ).to(device)

    decoder = Decoder(
        y_dim=train_buffer.y_dim,
        a_dim=config.a_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)

    dynamics_model = Dynamics(
        x_dim=config.x_dim,
        u_dim=train_buffer.u_dim,
        a_dim=config.a_dim,
        hidden_dim=config.hidden_dim,
        min_var=config.min_var,
        max_var=config.max_var,
        locally_linear=config.locally_linear,
    ).to(device)

    wandb.watch([encoder, dynamics_model, decoder], log="all", log_freq=10)

    all_params = (
        list(encoder.parameters()) +
        list(decoder.parameters()) +
        list(dynamics_model.parameters())
    )

    optimizer = torch.optim.Adam(all_params, lr=config.lr, eps=config.eps, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config.num_updates
    )

    # train and test loop
    for update in tqdm(range(config.num_updates)):

        # train
        encoder.train()
        decoder.train()
        dynamics_model.train()

        y, u, _, _ = train_buffer.sample(
            batch_size=config.batch_size,
            chunk_length=config.chunk_length,
        )

        y = torch.as_tensor(y, device=device)
        y = einops.rearrange(y, "b l y -> l b y")
        u = torch.as_tensor(u, device=device)
        u = einops.rearrange(u, "b l u -> l b u")

        losses = compute_elbo_losses(encoder, decoder, dynamics_model, y, u, config)

        optimizer.zero_grad()
        losses["total_loss"].backward()

        clip_grad_norm_(all_params, config.clip_grad_norm)
        optimizer.step()
        scheduler.step()

        wandb.log({
            "train/recon_loss": losses["recon_loss"].item(),
            "train/dynamics_kl": losses["dynamics_kl"].item(),
            "train/emission_loss": losses["emission_loss"].item(),
            "train/pred_loss": losses["pred_loss"].item(),
            "train/total_loss": losses["total_loss"].item(),
            "global_step": update,
        })

        if update % config.test_interval == 0:
            with torch.no_grad():
                encoder.eval()
                decoder.eval()
                dynamics_model.eval()

                y, u, _, _ = test_buffer.sample(
                    batch_size=config.batch_size,
                    chunk_length=config.chunk_length,
                )

                y = torch.as_tensor(y, device=device)
                y = einops.rearrange(y, "b l y -> l b y")
                u = torch.as_tensor(u, device=device)
                u = einops.rearrange(u, "b l u -> l b u")

                losses = compute_elbo_losses(encoder, decoder, dynamics_model, y, u, config)

                wandb.log({
                    "test/recon_loss": losses["recon_loss"].item(),
                    "test/dynamics_kl": losses["dynamics_kl"].item(),
                    "test/emission_loss": losses["emission_loss"].item(),
                    "test/pred_loss": losses["pred_loss"].item(),
                    "test/total_loss": losses["total_loss"].item(),
                    "global_step": update,
                })

    return encoder, decoder, dynamics_model


def train_cost(
    config: DictConfig,
    encoder: Encoder,
    dynamics_model: Dynamics,
    train_buffer: ReplayBuffer,
    test_buffer: ReplayBuffer,
):
    device = "cuda" if (torch.cuda.is_available() and not config.disable_gpu) else "cpu"

    cost_model = CostModel(
        x_dim=dynamics_model.x_dim,
        u_dim=dynamics_model.u_dim,
    ).to(device)

    # freeze backbone models
    for p in encoder.parameters():
        p.requires_grad = False

    for p in dynamics_model.parameters():
        p.requires_grad = False

    encoder.eval()
    dynamics_model.eval()

    wandb.watch([cost_model], log="all", log_freq=10)

    all_params = list(cost_model.parameters())
    optimizer = torch.optim.Adam(all_params, lr=config.lr, eps=config.eps, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config.num_updates
    )

    # train and test loop
    for update in tqdm(range(config.num_updates)):
        # train
        cost_model.train()

        y, u, c, _ = train_buffer.sample(
            batch_size=config.batch_size,
            chunk_length=config.chunk_length,
        )

        y = torch.as_tensor(y, device=device)
        y = einops.rearrange(y, "b l y -> l b y")
        # Use encoder mean for cost training (no sampling needed)
        a = encoder(einops.rearrange(y, "l b y -> (l b) y")).mean
        a = einops.rearrange(a, "(l b) a -> l b a", b=config.batch_size)
        u = torch.as_tensor(u, device=device)
        u = einops.rearrange(u, "b l u -> l b u")
        c = torch.as_tensor(c, device=device)
        c = einops.rearrange(c, "b l 1 -> l b 1")

        _, posteriors = dynamics_model(a=a, u=u)  # x0:T-1
        # compute cost loss
        cost_loss = nn.MSELoss()(
            cost_model(x=bottle_mvn(posteriors).loc, u=einops.rearrange(u, "l b u -> (l b) u")),
            einops.rearrange(c, "l b 1 -> (l b) 1")
        )
        optimizer.zero_grad()
        cost_loss.backward()

        clip_grad_norm_(all_params, config.clip_grad_norm)
        optimizer.step()
        scheduler.step()

        wandb.log({
            "train/cost loss": cost_loss.item(),
            "global_step": update,
        })

        if update % config.test_interval == 0:
            # test
            with torch.no_grad():
                cost_model.eval()

                y, u, c, _ = test_buffer.sample(
                    batch_size=config.batch_size,
                    chunk_length=config.chunk_length,
                )

                y = torch.as_tensor(y, device=device)
                y = einops.rearrange(y, "b l y -> l b y")
                a = encoder(einops.rearrange(y, "l b y -> (l b) y")).mean
                a = einops.rearrange(a, "(l b) a -> l b a", b=config.batch_size)
                u = torch.as_tensor(u, device=device)
                u = einops.rearrange(u, "b l u -> l b u")
                c = torch.as_tensor(c, device=device)
                c = einops.rearrange(c, "b l 1 -> l b 1")

                _, posteriors = dynamics_model(a=a, u=u)  # x0:T-1
                cost_loss = nn.MSELoss()(
                    cost_model(x=bottle_mvn(posteriors).loc, u=einops.rearrange(u, "l b u -> (l b) u")),
                    einops.rearrange(c, "l b 1 -> (l b) 1")
                )

                wandb.log({
                    "test/cost loss": cost_loss.item(),
                    "global_step": update,
                })

    return cost_model
