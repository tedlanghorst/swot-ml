from typing import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

from data import GraphBatch


def compute_loss(
    diff_model: PyTree,
    static_model: PyTree | None,
    batch: GraphBatch,
    norm_fn: Callable,
    denorm_fn: Callable,
    key: PRNGKeyArray,
    *,
    target_weights: dict[str, int],
    ood_lambda: float = 1.0,
    corr_keys: dict = {},
    loss_name: str = "mse",
) -> float:
    """Compute the loss between the predicted and true values.

    Parameters
    ----------
    diff_model: PyTree
        Differential components of the model.
    static_model: PyTree
        Static components of the model.
    batch: GraphBatch
        GraphBatch of data to use for training.
    keys: list[PRNGKeyArray]
        Batch of keys to use for the random dropout in the model.
    loss_name: str, optional
        Name of the loss function to use.
    target_weights: dict[str: float]
        Weights to use for the target variables.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    loss : float
    """
    model = eqx.combine(diff_model, static_model) if static_model is not None else diff_model
    y_hat = model(batch, key)

    if hasattr(batch, "node_mask"):
        data_mask = batch.node_mask
        if loss_name in ["nse", "spin_up_nse"]:
            # Expand the mask with a time dimension for broadcasting.
            data_mask = data_mask[jnp.newaxis, :]
    elif hasattr(batch, "d_mask"):
        data_mask = batch.d_mask
    else:
        raise AttributeError(f"Cannot find mask attribute in data batch:\n{batch.keys()=}")

    # Loop through the targets and calculate loss
    losses = []
    loss_fn = get_loss_fn(loss_name)
    for target_name, y_target in batch.y.items():
        # Skip targets the model didn't predict (or aren't in target_weights)
        if target_name not in y_hat:
            continue
        if target_weights.get(target_name, 0.0) == 0.0:
            continue

        y_hat_target = y_hat[target_name]

        if y_target.ndim > data_mask.ndim:
            y_target = y_target[..., 0]

        valid_mask = ~jnp.isnan(y_target) & data_mask
        masked_y = jnp.where(valid_mask, y_target, 0)

        # Handle GMM (dict output) vs Standard (Array output)
        if loss_name == "gmm_nll":
            expanded_mask = jnp.expand_dims(valid_mask, axis=-1)
            masked_y_hat = {
                "mu": jnp.where(expanded_mask, y_hat_target["mu"], 0.0),
                "sigma": jnp.where(expanded_mask, y_hat_target["sigma"], 1.0),  # sigma=1 is safe
                "log_pi": jnp.where(
                    expanded_mask, y_hat_target["log_pi"], 1.0 / y_hat_target["log_pi"].shape[-1]
                ),  # uniform is safe
            }
        elif loss_name == "manning_nll":
            masked_y_hat = y_hat_target
            masked_y_hat["Q"] = jnp.where(valid_mask[:, None, :], y_hat_target["Q"], 0)
        else:
            masked_y_hat = jnp.where(valid_mask, y_hat_target, 0)

        target_norm_fn = partial(norm_fn, name=target_name)
        target_denorm_fn = partial(denorm_fn, name=target_name)
        loss = loss_fn(masked_y, masked_y_hat, valid_mask, target_norm_fn, target_denorm_fn)

        has_any_valid = jnp.any(valid_mask)
        loss = jnp.where(has_any_valid, loss, 0.0)

        losses.append(loss)

    loss = jnp.mean(jnp.stack(losses))

    # Check for out distribution (ood) scores
    if "vae_ood_score" in y_hat:
        site_scores = y_hat["vae_ood_score"][jnp.newaxis, :]
        ood_score = jnp.mean(site_scores, where=data_mask)
        loss += ood_lambda * ood_score

    # Check for Attention Supervision
    if "attention_weights" in y_hat:
        # Re-use the data_mask for attention
        attn_loss_val = attention_leakage_loss(
            y_hat["attention_weights"],
            y_hat["valid_obs"],
            data_mask,
        )
        loss += attn_loss_val

    if corr_keys:
        x = y_hat[corr_keys['x']]
        y = y_target[corr_keys['y']]

        valid_mask = ~jnp.isnan(y) & data_mask
        masked_x = jnp.where(valid_mask, x, 0)
        masked_y = jnp.where(valid_mask, y, 0)

        reg_term = correlation_penalty(masked_x, masked_y, valid_mask)
        loss = loss + (corr_keys['lambda'] * reg_term)

    return loss


def get_loss_fn(loss_name):
    match loss_name:
        case "mse":
            return mse_loss
        case "mae":
            return mae_loss
        case "huber":
            return huber_loss
        case "nse":
            return nse_loss
        case "spin_up_nse":
            return spin_up_nse_loss
        case "gmm_nll":
            return gmm_nll
        case "manning_nll":
            return manning_nll
        case _:
            raise ValueError(f"Config'd loss ({loss_name}) not found.")


def mse_loss(y: Array, y_hat: Array, mask: Array, norm_fn: Callable, denorm_fn: Callable):
    """Calculates Mean Squared Error (MSE) loss."""
    mse = jnp.mean(jnp.square(y - y_hat), where=mask)
    return mse


def mae_loss(y: Array, y_hat: Array, mask: Array, norm_fn: Callable, denorm_fn: Callable):
    """Calculates Mean Absolute Error (MAE) loss."""
    mae = jnp.mean(jnp.abs(y - y_hat), where=mask)
    return mae


def huber_loss(y: Array, y_hat: Array, mask: Array, norm_fn: Callable, denorm_fn: Callable):
    """Calculates Huber loss."""
    # Passing delta through make_step is not yet implemented.
    huber_delta = 1

    residual = y - y_hat
    condition = jnp.abs(residual) <= huber_delta
    squared_loss = 0.5 * jnp.square(residual)
    linear_loss = huber_delta * (jnp.abs(residual) - 0.5 * huber_delta)
    return jnp.mean(jnp.where(condition, squared_loss, linear_loss), where=mask)


def nse_loss(y: Array, y_hat: Array, mask: Array, norm_fn: Callable, denorm_fn: Callable):
    """
    Calculates the smooth-joint NSE from Kratzert et al., 2019
    https://doi.org/10.5194/hess-23-5089-2019
    """
    sq_error = jnp.square(y - y_hat) * mask
    mse = jnp.mean(sq_error, axis=0)  # mean squared errors per basin

    std_y = jnp.std(y, axis=0, where=mask.astype(bool))  # Per-basin standard deviation
    stable_std_y = jnp.nan_to_num(std_y) + 0.1
    denom = jnp.square(stable_std_y)

    node_nse = mse / denom
    jnp.mean(node_nse, where=node_nse != 0)


def spin_up_nse_loss(y: Array, y_hat: Array, mask: Array, norm_fn: Callable, denorm_fn: Callable):
    """Equivalent to nse_loss but with weights that favor the final ~1/3 of the sequence"""
    y = denorm_fn(y)
    y_hat = denorm_fn(y_hat)

    seq_len = y.shape[0]
    weights = jax.nn.sigmoid(jnp.linspace(-10, 10, seq_len))

    sq_error = jnp.square(y - y_hat) * mask
    mse = jnp.average(sq_error, axis=0, weights=weights)  # mean squared errors per basin

    std_y = jnp.std(y, axis=0, where=mask.astype(bool))  # Per-basin standard deviation
    stable_std_y = jnp.nan_to_num(std_y) + 0.1
    denom = jnp.square(stable_std_y)

    node_nse = mse / denom
    return jnp.mean(node_nse, where=node_nse != 0)


def gmm_nll(
    y: Array, y_hat: dict[str, Array], mask: Array, norm_fn: Callable, denorm_fn: Callable
) -> Array:
    y_obs = jnp.expand_dims(y, axis=-1)
    mu, sigma, log_pi = y_hat["mu"], y_hat["sigma"], y_hat["log_pi"]

    log_sigma = jnp.log(sigma)
    log_2pi = jnp.log(2.0 * jnp.pi)

    # Calculate z-score: (obs - pred) / std
    scaled_diff = (y_obs - mu) / sigma
    # Cap the squared residual to prevent a single outlier from exploding gradients
    sq_diff = jnp.square(jnp.clip(scaled_diff, -20.0, 20.0))

    log_phi = -log_sigma - 0.5 * log_2pi - 0.5 * sq_diff

    # Numerical stability: logsumexp avoids pi * exp(log_phi) overflow
    log_likelihood = jax.scipy.special.logsumexp(log_pi + log_phi, axis=-1)

    return -jnp.sum(log_likelihood * mask) / (jnp.sum(mask) + 1e-8)


def manning_nll(
    y: Array, y_hat: dict[str, Array], mask: Array, norm_fn: Callable, denorm_fn: Callable
):
    y_hat_norm = norm_fn(y_hat["Q"])  # log(Q_samples), (B, S, R)
    mu_q = jnp.mean(y_hat_norm, axis=1)  # (B, R)
    var_q = jnp.mean((y_hat_norm - mu_q[:, None]) ** 2, axis=1)
    std_q = jnp.sqrt(var_q + 0.2**2)

    nll = jnp.log(std_q) + 0.5 * ((y - mu_q) / std_q) ** 2

    # Jacobian correction for log transform: -E[log Q] in physical space
    # y is already log-normalized, so exp(y) = Q_observed
    jacobian_correction = -jnp.mean(y_hat_norm, axis=1)  # -E[log Q_pred]
    nll = nll + jacobian_correction  # (B, R) — but wait, see note below

    valid = mask & jnp.isfinite(nll)
    return jnp.mean(nll, where=valid)


def attention_leakage_loss(attn_weights: Array, obs_mask: Array, node_mask: Array) -> Array:
    """
    Penalizes attention to masked (invalid) time steps AND encourages
    attention to valid time steps when they exist.
    """
    # Original penalty: don't attend to null token when real data exists
    real_data_exists = jnp.any(obs_mask[..., :-1], axis=-1)
    null_attn_weights = attn_weights[..., -1]
    leakage_penalty = null_attn_weights * real_data_exists

    # Sum of attention weights on valid time steps
    valid_attn_weights = attn_weights[..., :-1]  # Exclude null token
    valid_attn_sum = jnp.sum(valid_attn_weights * obs_mask[..., :-1], axis=-1)

    # Penalize low attention to valid data (when valid data exists)
    # Could use: 1 - valid_attn_sum to reward higher total attention on valid data
    underutilization_penalty = (1.0 - valid_attn_sum) * real_data_exists

    # Combine both penalties
    total_penalty = leakage_penalty + underutilization_penalty

    return jnp.mean(total_penalty, where=real_data_exists)


def correlation_penalty(y: Array, y_hat: Array, mask: Array) -> Array:
    """
    Calculates Pearson correlation penalty (|1-r|) over valid sequences.
    Inputs bypass denormalization to constrain execution within normalized space.
    """
    valid_count = jnp.sum(mask, axis=0)
    safe_count = jnp.where(valid_count > 0, valid_count, 1.0)
    
    mean_y = jnp.sum(y * mask, axis=0) / safe_count
    mean_y_hat = jnp.sum(y_hat * mask, axis=0) / safe_count
    
    y_centered = (y - mean_y) * mask
    y_hat_centered = (y_hat - mean_y_hat) * mask
    
    cov = jnp.sum(y_centered * y_hat_centered, axis=0)
    std_y = jnp.sqrt(jnp.sum(jnp.square(y_centered), axis=0))
    std_y_hat = jnp.sqrt(jnp.sum(jnp.square(y_hat_centered), axis=0))
    
    denominator = std_y * std_y_hat
    r = cov / (denominator + 1e-8)
    
    node_loss = jnp.abs(1.0 - r)
    
    return jnp.mean(node_loss, where=(valid_count > 1))


def mass_conservation_loss(
    S_pred: Array,
    Q_pred: Array,
    F_pred: Array,
    inflow_indices: Array,
    inflow_mask: Array,
    node_mask: Array,
    dt_seconds: float,
    fudge_lambda: float,
) -> Array:
    """
    Computes mass conservation regularization using network topology.
    Assumes inputs are denormalized arrays of shape (T, V).
    inflow_indices and inflow_mask have shape (V, max_parents).
    """
    # Calculate discrete time derivative for storage (T-1, V)
    dS = S_pred[1:] - S_pred[:-1]

    # Vectorized upstream index gathering (T, V, max_parents)
    Q_upstream = Q_pred[:, inflow_indices]
    
    # Isolate valid topological edges and sum incoming fluxes (T, V)
    Q_in = jnp.sum(jnp.where(inflow_mask, Q_upstream, 0.0), axis=-1)
    
    # Align flux arrays with the temporal bounds of dS
    Q_in_step = Q_in[1:]
    Q_out_step = Q_pred[1:]
    F_step = F_pred[1:]
    
    # Compute physical mass balance residual
    expected_dS = (Q_in_step - Q_out_step) * dt_seconds + F_step
    residual = dS - expected_dS
    
    valid_nodes = node_mask[jnp.newaxis, :]
    
    # Objective 1: Minimize closure error (L2)
    balance_loss = jnp.mean(jnp.square(residual), where=valid_nodes)
    
    # Objective 2: Suppress non-physical abstractions (L1 promotes sparsity)
    fudge_penalty = jnp.mean(jnp.abs(F_step), where=valid_nodes)
    
    return balance_loss + (fudge_lambda * fudge_penalty)
