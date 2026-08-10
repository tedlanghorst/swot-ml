import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from jax.scipy.special import logsumexp

from .base_model import BaseModel
from data import TabularBatch


class MixtureHead(eqx.Module):
    net: eqx.nn.MLP
    num_components: int

    def __init__(self, in_size, hidden_size, depth, num_components, *, key):
        self.num_components = num_components
        # Outputs: num_components * (weight_logit, mean, log_std)
        self.net = eqx.nn.MLP(
            in_size=in_size,
            out_size=num_components * 3,
            width_size=hidden_size,
            depth=depth,
            key=key,
        )

    def __call__(self, x):
        out = self.net(x)
        logits, means, log_stds = jnp.split(out, 3, axis=-1)
        log_weights = jax.nn.log_softmax(logits, axis=-1)  # (K,)
        log_stds = jnp.clip(log_stds, -6, 0)  # numerical stability
        return log_weights, means, log_stds

    def sample(self, x, key):
        log_weights, means, log_stds = self(x)
        k1, k2 = jax.random.split(key)
        # Gumbel-max trick to sample component
        component = jnp.argmax(log_weights + jax.random.gumbel(k1, shape=log_weights.shape))
        mean = means[component]
        std = jnp.exp(log_stds[component])
        return mean + std * jax.random.normal(k2, shape=mean.shape)

    def mean(self, x):
        """E[x] = sum_k w_k * mu_k"""
        log_weights, means, _ = self(x)
        return jnp.sum(jnp.exp(log_weights) * means)

    def log_prob(self, x, value):
        """Log likelihood of a scalar value under the mixture."""
        log_weights, means, log_stds = self(x)
        stds = jnp.exp(log_stds)
        log_component_probs = (
            -0.5 * ((value - means) / stds) ** 2 - log_stds - 0.5 * jnp.log(2 * jnp.pi)
        )
        return logsumexp(log_weights + log_component_probs)


class ManningNet(BaseModel):
    A0_head: MixtureHead
    n_head: MixtureHead
    num_components: int
    num_samples: int

    def __init__(
        self,
        *,
        target: str,
        dynamic_size: int,
        static_size: int,
        hidden_size: int,
        depth: int,
        num_components: int,
        num_samples: int,
        seed: int,
        **kwargs,  # Capture some defaults that we don't need.
    ):
        key = jax.random.PRNGKey(seed)
        k1, k2, k3 = jax.random.split(key, 3)
        super().__init__(hidden_size, target, "none", key=k1)

        self.num_components = num_components
        self.num_samples = num_samples
        self.A0_head = MixtureHead(
            dynamic_size + static_size, hidden_size, depth, num_components, key=k2
        )
        self.n_head = MixtureHead(
            dynamic_size + static_size, hidden_size, depth, num_components, key=k3
        )

    def __call__(self, batch: TabularBatch, key: PRNGKeyArray):
        batch_size = batch.x_d.shape[0]
        keys = jax.random.split(key, batch_size)

        return jax.vmap(self.single_reach)(batch, keys)

    def single_reach(self, batch: TabularBatch, key: PRNGKeyArray):
        def get_gmm_params(x_temporal):
            feat = jnp.concat([x_temporal, batch.x_s])
            return self.A0_head(feat), self.n_head(feat)

        all_A0_gmm, all_n_gmm = jax.vmap(get_gmm_params)(batch.x_d)

        k1, k2 = jax.random.split(key)
        A0_samples = self._sample_gmm_all(all_A0_gmm, k1)  # (T, S)
        n_samples = self._sample_gmm_all(all_n_gmm, k2)  # (T, S)

        A0_latent = jnp.mean(A0_samples, axis=0, where=batch.d_mask[:, None])
        n_latent = jnp.mean(n_samples, axis=0, where=batch.d_mask[:, None])

        # extract hws and safeguard the exponentiation from NaN by clipping.
        dh, w, s = batch.hws[:, 0], batch.hws[:, 1], batch.hws[:, 2]
        w = jnp.clip(w, 1e-6, None)
        s = jnp.clip(s, 0.0, None)

        dA = w * dh
        avg_w = jnp.mean(w, axis=0, where=batch.d_mask)
        min_dA = jnp.min(dA)  # can be -inf if dA has bad values
        min_dA = jnp.where(jnp.isfinite(min_dA), min_dA, 0.0)

        def manning(a0_latent, n_latent):
            # Clamp before any nonlinear transform
            a0_latent = jnp.clip(a0_latent, -5.0, 5.0)
            n_latent = jnp.clip(n_latent, -5.0, 5.0)
            A0 = (jax.nn.softplus(a0_latent) * avg_w) + 1e-6
            A = A0 + dA
            # Guard negative areas before fractional power
            A = jnp.clip(A, 1e-6, None)
            n = 0.3 + 0.299 * jnp.tanh(n_latent)
            Q = (1 / n) * jnp.power(A, 5 / 3) * jnp.power(w, -2 / 3) * jnp.sqrt(s)
            # Final guard — don't let Q explode into the loss
            Q = jnp.clip(Q, 0.0, 1e8)
            return Q, A0, n

        Q_samples, A0_samples, n_samples = jax.vmap(manning)(A0_latent, n_latent)

        return {self.target[0]: {"Q": Q_samples, "A0": A0_samples, "n": n_samples}}

    def _sample_gmm_all(self, gmm_params, key):
        """Sample S values from each of T GMMs. Returns (T, S)."""
        log_weights, means, log_stds = gmm_params  # each (T, K)
        T = means.shape[0]
        keys = jax.random.split(key, T)

        def sample_one(lw, mu, ls, k):
            # Draw S samples from a single K-component GMM
            k1, k2 = jax.random.split(k)
            # (S,) component indices via Gumbel-max
            gumbels = jax.random.gumbel(k1, shape=(self.num_samples, self.num_components))
            components = jnp.argmax(lw[None, :] + gumbels, axis=-1)  # (S,)
            # Gather means and stds for selected components
            sel_mu = mu[components]  # (S,)
            sel_std = jnp.exp(ls[components])  # (S,)
            samples = sel_mu + sel_std * jax.random.normal(k2, shape=(self.num_samples,))
            # Clamp latent samples to a sane range before physics transforms
            return jnp.clip(samples, -5.0, 5.0)

        return jax.vmap(sample_one)(log_weights, means, log_stds, keys)  # (T, S)

    def summarize_predictions(self, preds, d_mask):
        import numpy as np

        q_mean = np.median(preds["Q"], axis=-2)
        q_low = np.percentile(preds["Q"], 5, axis=-2)
        q_high = np.percentile(preds["Q"], 95, axis=-2)
        a0_est = np.median(preds["A0"], axis=-1)
        n_est = np.median(preds["n"], axis=-1)

        num_obs = preds["Q"].shape[-1]
        ravel_mask_idx = np.where(d_mask.ravel())[0]
        q_mean_flat = q_mean.ravel()[ravel_mask_idx]
        q_low_flat = q_low.ravel()[ravel_mask_idx]
        q_high_flat = q_high.ravel()[ravel_mask_idx]
        a0_flat = np.tile(a0_est[:, None], num_obs).ravel()[ravel_mask_idx]
        n_flat = np.tile(n_est[:, None], num_obs).ravel()[ravel_mask_idx]

        return {
            "Q_mean": q_mean_flat,
            "Q_05": q_low_flat,
            "Q_95": q_high_flat,
            "A0": a0_flat,
            "n": n_flat,
        }

    def stack_obs(self, obs_arr, d_mask):
        import numpy as np

        ravel_mask_idx = np.where(d_mask.ravel())[0]
        return obs_arr.ravel()[ravel_mask_idx]
