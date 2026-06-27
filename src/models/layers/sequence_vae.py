import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

class SequenceVAE(eqx.Module):
    encoder: eqx.nn.LSTMCell
    mu_proj: eqx.nn.Linear
    logvar_proj: eqx.nn.Linear
    decoder: eqx.nn.LSTMCell
    out_proj: eqx.nn.Linear
    hidden_size: int

    def __init__(self, input_dim: int, hidden_size: int, latent_dim: int, *, key: PRNGKeyArray):
        k1, k2, k3, k4, k5 = jrandom.split(key, 5)
        self.hidden_size = hidden_size
        
        self.encoder = eqx.nn.LSTMCell(input_dim, hidden_size, key=k1)
        
        self.mu_proj = eqx.nn.Linear(hidden_size, latent_dim, key=k2)
        self.logvar_proj = eqx.nn.Linear(hidden_size, latent_dim, key=k3)
        
        self.decoder = eqx.nn.LSTMCell(latent_dim, hidden_size, key=k4)
        self.out_proj = eqx.nn.Linear(hidden_size, input_dim, key=k5)

    def __call__(self, x_seq: Array, key: PRNGKeyArray) -> tuple[Array, Array, Array]:
        seq_length, _ = x_seq.shape
        
        def encode_step(state, x_t):
            new_state = self.encoder(x_t, state)
            return new_state, None
        
        init_enc_h = jnp.zeros((self.hidden_size,))
        init_enc_c = jnp.zeros((self.hidden_size,))
        final_enc_state, _ = jax.lax.scan(encode_step, (init_enc_h, init_enc_c), x_seq)

        final_h, _ = final_enc_state
        
        mu = self.mu_proj(final_h)
        logvar = self.logvar_proj(final_h)
        
        std = jnp.exp(0.5 * logvar)
        z = mu + jrandom.normal(key, std.shape) * std

        def decode_step(state, _):
            new_state = self.decoder(z, state)
            h_t, _ = new_state
            x_hat = self.out_proj(h_t)
            return new_state, x_hat
        
        init_dec_h = jnp.zeros((self.hidden_size,))
        init_dec_c = jnp.zeros((self.hidden_size,))
        _, recon_seq = jax.lax.scan(decode_step, (init_dec_h, init_dec_c), None, length=seq_length)

        return recon_seq, mu, logvar