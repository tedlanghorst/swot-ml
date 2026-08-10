import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


def segment_softmax(
    scores: Array, segment_ids: Array, num_segments: int, mask: Array
) -> Array:
    """Applies a numerically stable softmax to scores over segments with explicit masking."""
    safe_scores = jnp.where(mask, scores, -1e9)

    has_valid = (
        jax.ops.segment_max(mask.astype(jnp.float32), segment_ids, num_segments=num_segments)
        > 0.5
    )

    max_scores = jax.ops.segment_max(safe_scores, segment_ids, num_segments=num_segments)
    safe_max = jnp.where(has_valid, max_scores, 0.0)

    stable_scores = safe_scores - safe_max[segment_ids]
    exp_scores = jnp.exp(stable_scores) * mask
    sum_exp = jax.ops.segment_sum(exp_scores, segment_ids, num_segments=num_segments)

    result = exp_scores / jnp.maximum(sum_exp[segment_ids], 1e-8)
    result = jnp.where(has_valid[segment_ids], result, 0.0)

    return result


class MultiHeadFwdAttentionLayer(eqx.Module):
    """Computes forward attention using a fused MLP for all heads."""

    num_heads: int
    head_size: int
    attention_mlp: eqx.nn.MLP
    output_proj: eqx.nn.Linear

    def __init__(
        self,
        num_heads: int,
        node_hidden_size: int,
        static_feature_size: int,
        *,
        key: PRNGKeyArray,
    ):
        assert node_hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        mlp_key, proj_key = jax.random.split(key)

        self.num_heads = num_heads
        self.head_size = node_hidden_size // num_heads

        mlp_input_size = (node_hidden_size * 2) + (static_feature_size * 2)
        self.attention_mlp = eqx.nn.MLP(
            in_size=mlp_input_size,
            out_size=self.num_heads,
            width_size=node_hidden_size * 2,
            depth=1,
            use_bias=False,
            key=mlp_key,
        )
        self.output_proj = eqx.nn.Linear(
            node_hidden_size, node_hidden_size, use_bias=False, key=proj_key
        )

    def __call__(
        self, h: Array, x_s: Array, edge_index: tuple[Array, Array], edge_mask: Array
    ) -> tuple[Array, Array]:
        num_nodes, hidden_size = h.shape
        src, dest = edge_index

        mlp_input = jnp.concatenate([h[src], h[dest], x_s[src], x_s[dest]], axis=-1)
        raw_scores = jax.vmap(self.attention_mlp)(mlp_input)

        raw_scores = jax.nn.leaky_relu(raw_scores)
        temperature = jnp.sqrt(jnp.float32(self.head_size))
        scaled_scores = raw_scores / temperature

        weights = jax.vmap(segment_softmax, in_axes=(1, None, None, None), out_axes=1)(
            scaled_scores, dest, num_nodes, edge_mask
        )

        h_reshaped = h.reshape(num_nodes, self.num_heads, self.head_size)
        weighted_messages = h_reshaped[src] * weights[:, :, None]

        aggregated_messages = jax.vmap(
            lambda msg: jax.ops.segment_sum(msg, dest, num_nodes), in_axes=1, out_axes=1
        )(weighted_messages)

        aggregated_messages = aggregated_messages.reshape(num_nodes, hidden_size)
        projected_messages = jax.vmap(self.output_proj)(aggregated_messages)

        return projected_messages, weights


class MultiHeadRevGatingLayer(eqx.Module):
    """Computes reverse gating using a fused MLP for all heads."""

    num_heads: int
    head_size: int
    gate_mlp: eqx.nn.MLP
    output_proj: eqx.nn.Linear

    def __init__(
        self,
        num_heads: int,
        node_hidden_size: int,
        static_feature_size: int,
        *,
        key: PRNGKeyArray,
    ):
        assert node_hidden_size % num_heads == 0, "node_hidden_size must be divisible by num_heads"
        mlp_key, proj_key = jax.random.split(key)

        self.num_heads = num_heads
        self.head_size = node_hidden_size // num_heads

        mlp_input_size = (node_hidden_size * 2) + (static_feature_size * 2)
        self.gate_mlp = eqx.nn.MLP(
            in_size=mlp_input_size,
            out_size=self.num_heads,
            width_size=node_hidden_size * 2,
            depth=1,
            use_bias=False,
            key=mlp_key,
        )
        self.output_proj = eqx.nn.Linear(
            node_hidden_size, node_hidden_size, use_bias=False, key=proj_key
        )

    def __call__(
        self, h: Array, x_s: Array, edge_index: tuple[Array, Array], edge_mask: Array
    ) -> tuple[Array, Array]:
        num_nodes, hidden_size = h.shape
        src, dest = edge_index[::-1]

        mlp_input = jnp.concatenate([h[src], h[dest], x_s[src], x_s[dest]], axis=-1)

        raw_scores = jax.vmap(self.gate_mlp)(mlp_input)
        gates = jax.nn.sigmoid(raw_scores)

        masked_gates = gates * edge_mask[:, jnp.newaxis]

        h_reshaped = h.reshape(num_nodes, self.num_heads, self.head_size)
        gated_messages = h_reshaped[src] * masked_gates[:, :, None]

        aggregated_messages = jax.vmap(
            lambda msg: jax.ops.segment_sum(msg, dest, num_nodes), in_axes=1, out_axes=1
        )(gated_messages)

        aggregated_messages = aggregated_messages.reshape(num_nodes, hidden_size)
        projected_messages = jax.vmap(self.output_proj)(aggregated_messages)

        return projected_messages, gates


class MultiHeadStackedGAT(eqx.Module):
    """
    Applies k_hops of multi-head FwdAttention and RevGating, then
    integrates the results with a GRU-style update.
    """

    k_hops: int
    fwd_layers: list[MultiHeadFwdAttentionLayer]
    rev_layers: list[MultiHeadRevGatingLayer]
    norm: eqx.nn.LayerNorm

    reset_gate_mlp: eqx.nn.MLP
    update_gate_mlp: eqx.nn.MLP
    candidate_mlp: eqx.nn.MLP

    def __init__(
        self,
        static_feature_size: int,
        hidden_size: int,
        k_hops: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
    ):
        self.k_hops = k_hops
        fwd_keys, rev_keys, gate_keys = jax.random.split(key, 3)
        fwd_keys = jax.random.split(fwd_keys, k_hops)
        rev_keys = jax.random.split(rev_keys, k_hops)
        r_key, z_key, c_key = jax.random.split(gate_keys, 3)

        self.fwd_layers = [
            MultiHeadFwdAttentionLayer(num_heads, hidden_size, static_feature_size, key=k)
            for k in fwd_keys
        ]
        self.rev_layers = [
            MultiHeadRevGatingLayer(num_heads, hidden_size, static_feature_size, key=k)
            for k in rev_keys
        ]
        self.norm = eqx.nn.LayerNorm(hidden_size)

        gate_input_size = hidden_size * 3
        self.reset_gate_mlp = eqx.nn.MLP(
            gate_input_size, hidden_size, hidden_size * 3, 1, key=r_key
        )
        self.update_gate_mlp = eqx.nn.MLP(
            gate_input_size, hidden_size, hidden_size * 3, 1, key=z_key
        )
        self.candidate_mlp = eqx.nn.MLP(
            gate_input_size, hidden_size, hidden_size * 3, 1, key=c_key
        )

    def __call__(
        self, x: Array, x_s: Array, edge_index: Array, edge_mask: Array
    ) -> tuple[Array, Array, Array, Array, Array]:
        fwd_ws, rev_ws = [], []

        h_fwd = x
        for layer in self.fwd_layers:
            h_norm = jax.vmap(self.norm)(h_fwd)
            fwd_messages, fwd_w = layer(h_norm, x_s, edge_index, edge_mask)
            h_fwd = h_fwd + fwd_messages
            fwd_ws.append(fwd_w)
        fwd_ws = jnp.stack(fwd_ws, axis=-1)

        h_rev = x
        for layer in self.rev_layers:
            h_norm = jax.vmap(self.norm)(h_rev)
            rev_messages, rev_w = layer(h_norm, x_s, edge_index, edge_mask)
            h_rev = h_rev + rev_messages
            rev_ws.append(rev_w)
        rev_ws = jnp.stack(rev_ws, axis=-1)

        fwd_aggregated = h_fwd - x
        rev_aggregated = h_rev - x

        messages = jnp.concatenate([fwd_aggregated, rev_aggregated], axis=-1)
        gate_input = jnp.concatenate([x, messages], axis=-1)

        r_gate = jax.nn.sigmoid(jax.vmap(self.reset_gate_mlp)(gate_input))
        z_gate = jax.nn.sigmoid(jax.vmap(self.update_gate_mlp)(gate_input))

        candidate_input = jnp.concatenate([r_gate * x, messages], axis=-1)
        candidate_state = jnp.tanh(jax.vmap(self.candidate_mlp)(candidate_input))

        final_h = (1 - z_gate) * x + z_gate * candidate_state

        return final_h, fwd_ws, rev_ws, z_gate, r_gate


class SpatioTemporalLSTMCell(eqx.Module):
    """
    A recurrent cell with separate temporal and spatial steps, allowing
    the caller to insert assimilation between them.

    Temporal step: LSTM updates c from input and previous state.
    Spatial step: GAT propagates c across the graph.
    """

    gat: MultiHeadStackedGAT
    lstm_cell: eqx.nn.LSTMCell
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        lstm_input_size: int,
        static_feature_size: int,
        hidden_size: int,
        k_hops: int,
        num_heads: int,
        dropout_p: float,
        *,
        key: PRNGKeyArray,
    ):
        gat_key, lstm_key = jax.random.split(key)

        self.gat = MultiHeadStackedGAT(
            static_feature_size=static_feature_size,
            hidden_size=hidden_size,
            k_hops=k_hops,
            num_heads=num_heads,
            key=gat_key,
        )
        self.lstm_cell = eqx.nn.LSTMCell(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            key=lstm_key,
        )
        self.dropout = eqx.nn.Dropout(dropout_p)

    def temporal_step(
        self,
        x_t: Array,
        state: tuple[Array, Array],
        node_mask: Array,
    ) -> Array:
        """
        LSTM temporal update only. Returns the new cell state c.

        Args:
            x_t: Input features at this timestep. (num_nodes, input_size)
            state: Tuple of (h_prev, c_prev).
            node_mask: Boolean mask for valid nodes. (num_nodes,)

        Returns:
            c_new: Updated cell state after temporal processing. (num_nodes, hidden_size)
        """
        h_prev, c_prev = state
        _, c_new = jax.vmap(self.lstm_cell)(x_t, (h_prev, c_prev))
        c_new = c_new * node_mask[:, jnp.newaxis]
        return c_new

    def spatial_step(
        self,
        c: Array,
        node_features: Array,
        edge_index: Array,
        node_mask: Array,
        edge_mask: Array,
        *,
        key: PRNGKeyArray,
    ) -> tuple[Array, dict]:
        """
        GAT spatial propagation on cell state c.

        Args:
            c: Cell state to propagate spatially. (num_nodes, hidden_size)
            node_features: Static node embeddings. (num_nodes, static_size)
            edge_index: Graph edge indices. (2, num_edges)
            node_mask: Boolean mask for valid nodes. (num_nodes,)
            edge_mask: Boolean mask for valid edges. (num_edges,)
            key: PRNG key for dropout.

        Returns:
            c_spatial: Spatially propagated cell state. (num_nodes, hidden_size)
            trace_data: Dict of attention weights and gates.
        """
        c_spatial, fwd_w, rev_w, z, r = self.gat(
            c, node_features, edge_index=edge_index, edge_mask=edge_mask
        )
        c_spatial = self.dropout(c_spatial, key=key)
        c_spatial = c_spatial * node_mask[:, jnp.newaxis]

        trace_data = {"forward_w": fwd_w, "reverse_w": rev_w, "z_gate": z, "r_gate": r}

        return c_spatial, trace_data