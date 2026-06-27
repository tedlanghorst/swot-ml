from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from data import GraphBatch
from .base_model import BaseModel
from .layers.static_mlp import StaticMLP
from .layers.sequence_vae import SequenceVAE


def segment_softmax(logits: Array, segment_ids: Array, num_segments: int) -> Array:
    """Compute softmax over segments safely."""
    m = jax.ops.segment_max(logits, segment_ids, num_segments)
    logits = logits - m[segment_ids]
    e = jnp.exp(logits)
    z = jax.ops.segment_sum(e, segment_ids, num_segments)
    return e / (z[segment_ids] + 1e-9)


class DownstreamAggregation(eqx.Module):
    """
    Sparse implementation: Aggregate information flowing downstream.
    Only computes attention on connected edges (src -> dest).
    """

    num_heads: int
    hidden_size: int
    head_dim: int

    query_proj: eqx.nn.Linear
    key_proj: eqx.nn.Linear
    value_proj: eqx.nn.Linear

    static_proj: eqx.nn.Linear
    norm: eqx.nn.LayerNorm
    out_proj: eqx.nn.Linear

    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        static_feature_size: int,
        dropout_p: float = 0.0,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, 5)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.query_proj = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=keys[0])
        self.key_proj = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=keys[1])
        self.value_proj = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=keys[2])

        self.static_proj = eqx.nn.Linear(
            static_feature_size, hidden_size, use_bias=False, key=keys[3]
        )
        self.out_proj = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=keys[4])
        self.norm = eqx.nn.LayerNorm(hidden_size)

    def __call__(
        self,
        c: Array,
        x_s: Array,
        edge_index: tuple[Array, Array],
        node_mask: Array,
        edge_mask: Array,
        *,
        key: PRNGKeyArray = None,
    ) -> tuple[Array, Array]:
        num_nodes = c.shape[0]
        src, dest = edge_index

        static_emb = jax.vmap(self.static_proj)(x_s)
        c_cond = jax.vmap(self.norm)(c + static_emb)

        Q = jax.vmap(self.query_proj)(c_cond).reshape(num_nodes, self.num_heads, self.head_dim)
        K = jax.vmap(self.key_proj)(c_cond).reshape(num_nodes, self.num_heads, self.head_dim)
        V = jax.vmap(self.value_proj)(c_cond).reshape(num_nodes, self.num_heads, self.head_dim)

        Q_dest = Q[dest]
        K_src = K[src]
        V_src = V[src]

        scale = jnp.sqrt(self.head_dim)
        scores = jnp.einsum("ehd,ehd->eh", Q_dest, K_src) / scale
        scores = jnp.where(edge_mask[:, None], scores, -1e9)

        attn_weights = segment_softmax(scores, dest, num_nodes)

        weighted_V = V_src * attn_weights[:, :, None]
        weighted_V_flat = weighted_V.reshape(weighted_V.shape[0], self.hidden_size)
        aggr_out = jax.ops.segment_sum(weighted_V_flat, dest, num_nodes)

        output = jax.vmap(self.out_proj)(aggr_out)
        output = output * node_mask[:, None]

        return output, attn_weights


class UpstreamPropagation(eqx.Module):
    """
    Propagate information upstream (against physical flow direction).
    Each node receives from exactly one downstream neighbor (if any).
    """

    hidden_size: int
    message_net: eqx.nn.MLP
    gate_net: eqx.nn.MLP

    def __init__(
        self,
        hidden_size: int,
        static_feature_size: int,
        *,
        key: PRNGKeyArray,
    ):
        k1, k2 = jax.random.split(key)

        self.hidden_size = hidden_size

        msg_input_size = hidden_size + static_feature_size
        self.message_net = eqx.nn.MLP(
            in_size=msg_input_size,
            out_size=hidden_size,
            width_size=hidden_size,
            depth=1,
            key=k1,
        )

        gate_input_size = hidden_size * 2 + static_feature_size * 2
        self.gate_net = eqx.nn.MLP(
            in_size=gate_input_size,
            out_size=hidden_size,
            width_size=hidden_size,
            depth=1,
            key=k2,
        )

    def __call__(
        self,
        c: Array,
        x_s: Array,
        edge_index: tuple[Array, Array],
        node_mask: Array,
        edge_mask: Array,
    ) -> tuple[Array, Array]:
        num_nodes = c.shape[0]
        src, dest = edge_index

        sender = dest
        receiver = src

        sender_input = jnp.concatenate([c[sender], x_s[sender]], axis=-1)
        message = jax.vmap(self.message_net)(sender_input)

        gate_input = jnp.concatenate([c[sender], c[receiver], x_s[sender], x_s[receiver]], axis=-1)
        gate = jax.nn.sigmoid(jax.vmap(self.gate_net)(gate_input))

        gated_message = message * gate * edge_mask[:, jnp.newaxis]

        output = jnp.zeros((num_nodes, self.hidden_size))
        output = output.at[receiver].add(gated_message)

        gate_per_node = jnp.zeros((num_nodes, self.hidden_size))
        gate_per_node = gate_per_node.at[receiver].set(gate * edge_mask[:, jnp.newaxis])

        return output, gate_per_node


class BidirectionalGraphLayer(eqx.Module):
    """
    Single layer of bidirectional message passing:
    - Downstream: attention-weighted aggregation from tributaries
    - Upstream: gated propagation from downstream neighbor
    """

    downstream_agg: DownstreamAggregation
    upstream_prop: UpstreamPropagation
    norm: eqx.nn.LayerNorm
    combine_proj: eqx.nn.Linear

    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        static_feature_size: int,
        dropout_p: float = 0.0,
        *,
        key: PRNGKeyArray,
    ):
        k1, k2, k3 = jax.random.split(key, 3)

        self.downstream_agg = DownstreamAggregation(
            num_heads=num_heads,
            hidden_size=hidden_size,
            static_feature_size=static_feature_size,
            dropout_p=dropout_p,
            key=k1,
        )

        self.upstream_prop = UpstreamPropagation(
            hidden_size=hidden_size,
            static_feature_size=static_feature_size,
            key=k2,
        )

        self.norm = eqx.nn.LayerNorm(hidden_size)
        self.combine_proj = eqx.nn.Linear(hidden_size * 2, hidden_size, key=k3)

    def __call__(
        self,
        c: Array,
        x_s: Array,
        edge_index: tuple[Array, Array],
        node_mask: Array,
        edge_mask: Array,
        *,
        key: PRNGKeyArray = None,
    ) -> tuple[Array, dict]:
        c_norm = jax.vmap(self.norm)(c)

        downstream_msg, downstream_weight = self.downstream_agg(
            c_norm, x_s, edge_index, node_mask, edge_mask, key=key
        )

        upstream_msg, upstream_gate = self.upstream_prop(
            c_norm, x_s, edge_index, node_mask, edge_mask
        )

        combined = jnp.concatenate([downstream_msg, upstream_msg], axis=-1)
        message = jax.vmap(self.combine_proj)(combined)

        output = c + message

        trace = {
            "downstream_weight": downstream_weight,
            "upstream_gate": upstream_gate,
        }

        return output, trace


class StackedGraphAttention(eqx.Module):
    """
    K-hop bidirectional graph processing with GRU-style output gating.
    """

    k_hops: int
    hidden_size: int
    layers: list[BidirectionalGraphLayer]

    reset_gate: eqx.nn.Linear
    update_gate: eqx.nn.Linear
    candidate_proj: eqx.nn.Linear

    def __init__(
        self,
        static_feature_size: int,
        hidden_size: int,
        k_hops: int,
        num_heads: int,
        dropout_p: float = 0.0,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, k_hops + 3)

        self.k_hops = k_hops
        self.hidden_size = hidden_size

        self.layers = [
            BidirectionalGraphLayer(
                num_heads=num_heads,
                hidden_size=hidden_size,
                static_feature_size=static_feature_size,
                dropout_p=dropout_p,
                key=keys[i],
            )
            for i in range(k_hops)
        ]

        gate_input_size = hidden_size * 2
        self.reset_gate = eqx.nn.Linear(gate_input_size, hidden_size, key=keys[-3])
        self.update_gate = eqx.nn.Linear(gate_input_size, hidden_size, key=keys[-2])
        self.candidate_proj = eqx.nn.Linear(gate_input_size, hidden_size, key=keys[-1])

    def __call__(
        self,
        x: Array,
        x_s: Array,
        edge_index: tuple[Array, Array],
        node_mask: Array,
        edge_mask: Array,
        *,
        key: PRNGKeyArray = None,
    ) -> tuple[Array, dict, Array, Array]:
        if len(self.layers) == 0:
            return x, {}, None, None

        keys = jax.random.split(key, self.k_hops) if key is not None else [None] * self.k_hops

        layer_traces = []

        h = x
        for layer, k in zip(self.layers, keys):
            h, trace = layer(h, x_s, edge_index, node_mask, edge_mask, key=k)
            layer_traces.append(trace)

        stacked_traces = {
            key: jnp.stack([t[key] for t in layer_traces], axis=-1)
            for key in layer_traces[0].keys()
        }

        messages = h - x
        gate_input = jnp.concatenate([x, messages], axis=-1)

        r = jax.nn.sigmoid(jax.vmap(self.reset_gate)(gate_input))
        z = jax.nn.sigmoid(jax.vmap(self.update_gate)(gate_input))

        candidate_input = jnp.concatenate([r * x, messages], axis=-1)
        candidate = jnp.tanh(jax.vmap(self.candidate_proj)(candidate_input))

        output = (1 - z) * x + z * candidate

        return output, stacked_traces, z, r


class SequentialKalmanAssimilation(eqx.Module):
    """
    Sequential Kalman-style data assimilation on the LSTM cell state.

    For each observation source s (processed in order):
        z'_{s,t} = H_s(z_{s,t})           [computed externally by SourceEmbedderRegistry]
        K_{s,t} = sigma(W_{s,k} . [h_{t-1}, z'_{s,t}, log(1+dt_{s,t})] + b_{s,k})
        c_t^s = c_t^{s-1} + K_{s,t} * (z'_{s,t} - c_t^{s-1})

    Each source's update sees the result of all prior sources' updates,
    making the assimilation order-dependent and sequential.
    """

    hidden_size: int
    source_names: list[str]
    gain_networks: dict[str, eqx.nn.Linear]

    def __init__(self, hidden_size: int, source_names: list[str], *, key: PRNGKeyArray):
        self.hidden_size = hidden_size
        self.source_names = list(source_names)
        self.gain_networks = {}

        # Gain input: [h_prev (H), z'_s (H), log(1+dt) (1)] -> gain (H)
        gain_input_size = hidden_size * 2 + 1
        keys = jax.random.split(key, max(len(source_names), 1))
        for i, name in enumerate(source_names):
            self.gain_networks[name] = eqx.nn.Linear(
                gain_input_size, hidden_size, key=keys[i]
            )

    def register_source(self, name: str, *, key: PRNGKeyArray):
        """Add a gain network for a new observation source."""
        gain_input_size = self.hidden_size * 2 + 1
        self.gain_networks[name] = eqx.nn.Linear(
            gain_input_size, self.hidden_size, key=key
        )
        self.source_names.append(name)

    def __call__(
        self,
        c: Array,
        h_prev: Array,
        obs_embeddings: dict[str, Array],
        obs_valid: dict[str, Array],
        obs_staleness: dict[str, Array],
    ) -> tuple[Array, dict]:
        """
        Sequentially assimilate each source into the cell state.

        Args:
            c: (N, H) forecasted cell state c_t'
            h_prev: (N, H) previous hidden state h_{t-1}
            obs_embeddings: {source: (N, H)} z'_{s,t} = H_s(z_{s,t})
            obs_valid: {source: (N,)} boolean mask (True = assimilate)
            obs_staleness: {source: (N,)} dt_{s,t} time since last obs

        Returns:
            c_updated: (N, H) assimilated cell state
            trace: dict with per-source gain arrays
        """
        gains_list = []

        for name in self.source_names:
            if name not in obs_embeddings:
                gains_list.append(jnp.zeros_like(c))
                continue

            z_prime = obs_embeddings[name]   # (N, H)
            valid = obs_valid[name]          # (N,)
            staleness = obs_staleness[name]  # (N,)

            # K_{s,t} = sigma(W_{s,k} . [h_{t-1}, z'_{s,t}, log(1+dt_{s,t})] + b_{s,k})
            log_dt = jnp.log1p(staleness)[:, None]  # (N, 1)
            gain_input = jnp.concatenate([h_prev, z_prime, log_dt], axis=-1)  # (N, 2H+1)
            K = jax.nn.sigmoid(jax.vmap(self.gain_networks[name])(gain_input))  # (N, H)

            # c_t^s = c_t^{s-1} + K_{s,t} * (z'_{s,t} - c_t^{s-1})
            innovation = z_prime - c
            update = K * innovation
            c = c + jnp.where(valid[:, None], update, 0.0)

            gains_list.append(K * valid[:, None])

        kalman_gains = jnp.stack(gains_list) if gains_list else jnp.zeros(
            (0, c.shape[0], self.hidden_size)
        )

        return c, {"kalman_gains": kalman_gains}


class SpatioTemporalLSTMCell(eqx.Module):
    """
    Modified LSTM cell with sequential Kalman assimilation and graph message passing
    applied to the memory state (c) before producing the hidden output (h).

    Processing order at each timestep:
        1. LSTM Forecast: compute gates and c_t' = f*c_{t-1} + i*c_tilde
        2. Sequential Kalman Assimilation on c_t' (per-source, in order)
        3. Bidirectional Graph Message Passing on assimilated c
        4. LSTM Output: h_t = o_t * tanh(c_final)

    This ensures h_t reflects all interventions on the cell state.
    """

    gat: StackedGraphAttention
    kalman: SequentialKalmanAssimilation
    hidden_size: int
    input_size: int

    # Manual LSTM weights for split forward pass
    weight_ih: Array  # (4*H, input_size)
    weight_hh: Array  # (4*H, H)
    bias: Array       # (4*H,)

    def __init__(
        self,
        lstm_input_size: int,
        static_feature_size: int,
        hidden_size: int,
        k_hops: int,
        num_heads: int,
        source_names: list[str],
        dropout_p: float,
        *,
        key: PRNGKeyArray,
    ):
        gat_key, kalman_key, lstm_key = jax.random.split(key, 3)

        self.hidden_size = hidden_size
        self.input_size = lstm_input_size

        self.gat = StackedGraphAttention(
            static_feature_size=static_feature_size,
            hidden_size=hidden_size,
            k_hops=k_hops,
            num_heads=num_heads,
            dropout_p=dropout_p,
            key=gat_key,
        )

        self.kalman = SequentialKalmanAssimilation(
            hidden_size=hidden_size,
            source_names=source_names,
            key=kalman_key,
        )

        # LSTM weights (Xavier uniform initialization)
        k1, k2 = jax.random.split(lstm_key)
        scale_ih = 1.0 / jnp.sqrt(lstm_input_size)
        scale_hh = 1.0 / jnp.sqrt(hidden_size)
        self.weight_ih = jax.random.uniform(
            k1, (4 * hidden_size, lstm_input_size), minval=-scale_ih, maxval=scale_ih
        )
        self.weight_hh = jax.random.uniform(
            k2, (4 * hidden_size, hidden_size), minval=-scale_hh, maxval=scale_hh
        )
        # Forget gate bias initialized to 1.0 for better gradient flow
        bias = jnp.zeros(4 * hidden_size)
        self.bias = bias.at[hidden_size : 2 * hidden_size].set(1.0)

    def _lstm_forecast(self, x: Array, h: Array, c: Array) -> tuple[Array, Array]:
        """
        LSTM forecast phase: compute gates and forecasted cell state.

        Gate layout: [input, forget, output, cell_candidate]

        Returns:
            c_prime: forecasted cell state (f * c_{t-1} + i * c_tilde)
            o: output gate (to be applied AFTER cell state modifications)
        """
        gates = self.weight_ih @ x + self.weight_hh @ h + self.bias
        i, f, o, g = jnp.split(gates, 4)

        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        o = jax.nn.sigmoid(o)
        g = jnp.tanh(g)

        c_prime = f * c + i * g
        return c_prime, o

    def __call__(
        self,
        x_t: Array,
        state: tuple[Array, Array],
        obs_embeddings: dict[str, Array],
        obs_valid: dict[str, Array],
        obs_staleness: dict[str, Array],
        node_features: Array,
        edge_index: tuple[Array, Array],
        node_mask: Array,
        edge_mask: Array,
        *,
        key: PRNGKeyArray,
    ) -> tuple[tuple[Array, Array], dict]:
        """
        Full forward pass of the spatio-temporal cell.

        Args:
            x_t: (N, input_size) dense input embedding at time t
            state: (h_prev, c_prev) each (N, H)
            obs_embeddings: {source: (N, H)} embedded observations z'_{s,t}
            obs_valid: {source: (N,)} boolean validity masks
            obs_staleness: {source: (N,)} time since last observation
            node_features: (N, static_size) static node attributes
            edge_index: (src, dest) edge arrays
            node_mask: (N,) valid node mask
            edge_mask: (E,) valid edge mask
            key: PRNG key for graph attention

        Returns:
            new_state: (h_new, c_new) updated states
            trace: dict with spatial and assimilation diagnostics
        """
        h_prev, c_prev = state

        # 1. LSTM Forecast Phase
        c_prime, o_t = jax.vmap(self._lstm_forecast)(x_t, h_prev, c_prev)
        c_prime = c_prime * node_mask[:, None]
        o_t = o_t * node_mask[:, None]

        # 2. Sequential Kalman Assimilation on c_prime
        c_assim, kalman_trace = self.kalman(
            c_prime, h_prev, obs_embeddings, obs_valid, obs_staleness
        )
        c_assim = c_assim * node_mask[:, None]

        # 3. Graph Message Passing on assimilated cell state
        c_spatial, spatial_trace, z, r = self.gat(
            c_assim,
            node_features,
            edge_index=edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
            key=key,
        )

        # 4. LSTM Output Phase: h_t = o_t * tanh(c_final)
        h_new = o_t * jnp.tanh(c_spatial)
        h_new = h_new * node_mask[:, None]

        new_state = (h_new, c_spatial)

        trace = {**spatial_trace, **kalman_trace}
        trace["z_gate"] = z
        trace["r_gate"] = r

        return new_state, trace


class SourceEmbedderRegistry(eqx.Module):
    """Registry of source-specific embedders (H_s operators) mapping to a common hidden space."""

    embedders: dict[str, StaticMLP]
    norms: dict[str, eqx.nn.LayerNorm]
    hidden_size: int
    static_size: int

    def __init__(self, hidden_size: int, static_size: int):
        self.embedders = {}
        self.norms = {}
        self.hidden_size = hidden_size
        self.static_size = static_size

    def register_source(
        self,
        name: str,
        n_features: int,
        embed_width: int = None,
        embed_depth: int = 2,
        *,
        key: PRNGKeyArray,
    ):
        embed_width = embed_width or self.hidden_size

        self.embedders[name] = StaticMLP(
            dynamic_in_size=n_features,
            static_in_size=self.static_size,
            out_size=self.hidden_size,
            width_size=embed_width,
            depth=embed_depth,
            key=key,
        )
        self.norms[name] = eqx.nn.LayerNorm(self.hidden_size)

    def embed(self, name: str, features: Array, static: Array) -> tuple[Array, Array]:
        obs_mask = ~jnp.any(jnp.isnan(features), axis=-1)
        safe_features = jnp.nan_to_num(features)

        raw_emb = self.embedders[name](safe_features, static)
        normed_emb = jax.vmap(self.norms[name])(raw_emb)
        masked_emb = jnp.where(obs_mask[:, jnp.newaxis], normed_emb, 0.0)

        return masked_emb, obs_mask

    @property
    def sources(self) -> list[str]:
        return list(self.embedders.keys())


class SparseFusionGRNN_C(BaseModel):
    """
    Spatio-Temporal Graph Recurrent Neural Network with:
    - Modified LSTM with split forecast/output phases
    - Sequential Kalman-style data assimilation on the cell state
    - Bidirectional graph message passing on the cell state
    - Modular source embedders (H_s operators) for observation types
    """

    backbone_leaves: list[str]
    hidden_size: int
    seq_length: int
    dense_sources: list[str]
    static_size: int
    seq2seq: bool
    supervised_attn: bool
    use_obs_memory: bool
    return_weights: bool
    staleness_threshold: float

    sequence_vae: SequenceVAE | None
    dense_embedders: dict[str, eqx.nn.MLP]
    fusion_norm: eqx.nn.LayerNorm
    source_registry: SourceEmbedderRegistry
    spat_temp_lstm: SpatioTemporalLSTMCell
    dropout: eqx.nn.Dropout

    _dense_sizes: list[int]
    _dense_offsets: list[int]

    def __init__(
        self,
        *,
        target: list,
        seq_length: int,
        dense_sizes: dict[str, int],
        sparse_sizes: dict[str, int],
        static_size: int,
        hidden_size: int,
        k_hops: int,
        num_heads: int,
        assim_sizes: dict[str, dict[str, int]],
        seed: int,
        dropout: float,
        head: str,
        seq2seq: bool,
        supervised_attn: bool,
        use_obs_memory: bool,
        return_weights: bool,
        staleness_threshold: float = 14.0,
        use_sequence_vae: bool = False,
    ):
        key = jrandom.PRNGKey(seed)
        keys = list(jrandom.split(key, 10))

        super().__init__(hidden_size, target, head, key=keys.pop(0))

        self.backbone_leaves = ["dense_embedders", "fusion_norm", "spat_temp_lstm", "head"]
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.dense_sources = list(dense_sizes.keys())
        self.static_size = static_size
        self.seq2seq = seq2seq
        self.supervised_attn = supervised_attn
        self.return_weights = return_weights
        self.use_obs_memory = use_obs_memory
        self.staleness_threshold = staleness_threshold

        self.dropout = eqx.nn.Dropout(dropout)

        # Optional VAE
        if use_sequence_vae:
            sum_dense = sum(dense_sizes.values())
            self.sequence_vae = SequenceVAE(sum_dense, sum_dense * 2, sum_dense, key=keys.pop(0))
        else:
            self.sequence_vae = None

        # Dense embedders
        dense_keys = jrandom.split(keys.pop(0), len(self.dense_sources))
        self.dense_embedders = {}
        for (name, size), k in zip(dense_sizes.items(), dense_keys):
            self.dense_embedders[name] = eqx.nn.MLP(
                in_size=size,
                out_size=hidden_size,
                width_size=hidden_size,
                depth=1,
                key=k,
            )
        self._dense_sizes = [dense_sizes[name] for name in self.dense_sources]
        self._dense_offsets = [0] + list(jnp.cumsum(jnp.array(self._dense_sizes[:-1])))
        self.fusion_norm = eqx.nn.LayerNorm(hidden_size)

        # Sparse observation embedders (H_s operators)
        self.source_registry = SourceEmbedderRegistry(hidden_size, static_size)
        sparse_keys = jrandom.split(keys.pop(0), len(sparse_sizes))
        for (name, in_size), k in zip(sparse_sizes.items(), sparse_keys):
            size_args = assim_sizes.get(name, {})
            self.source_registry.register_source(
                name,
                in_size,
                embed_width=size_args.get("embed_width"),
                embed_depth=size_args.get("embed_depth", 2),
                key=k,
            )

        # Spatio-temporal cell (includes Kalman assimilation + graph)
        self.spat_temp_lstm = SpatioTemporalLSTMCell(
            lstm_input_size=hidden_size,
            static_feature_size=static_size,
            hidden_size=hidden_size,
            k_hops=k_hops,
            num_heads=num_heads,
            source_names=list(sparse_sizes.keys()),
            dropout_p=dropout,
            key=keys.pop(0),
        )

    def add_assimilator(
        self,
        name: str,
        n_features: int,
        size_args: dict = {},
        *,
        key: PRNGKeyArray = None,
    ):
        """Add a new observation source (embedder + gain network)."""
        key = key if key is not None else jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)

        # Register the H_s embedder
        self.source_registry.register_source(
            name,
            n_features,
            embed_width=size_args.get("embed_width"),
            embed_depth=size_args.get("embed_depth", 2),
            key=k1,
        )

        # Register the gain network in the Kalman module
        self.spat_temp_lstm.kalman.register_source(name, key=k2)

    @property
    def sparse_sources(self) -> list[str]:
        return self.source_registry.sources

    def __call__(self, data: GraphBatch, key: PRNGKeyArray) -> Array:
        num_locations = data.static.shape[0]
        num_sources = len(self.source_registry.sources)
        node_mask = data.node_mask
        edge_mask = data.edge_mask
        graph_edges = data.graph_edges

        vae_key, drop_key1, drop_key2, scan_key = jrandom.split(key, 4)

        # 0. Optional VAE
        if self.sequence_vae is not None:
            raw_dense_list = [data.dynamic[name] for name in self.dense_sources]
            raw_dense_seq = jnp.concatenate(raw_dense_list, axis=-1)
            vae_keys = jrandom.split(vae_key, num_locations)
            recon_seq, mu, logvar = jax.vmap(
                self.sequence_vae, in_axes=(1, 0), out_axes=(1, 0, 0)
            )(raw_dense_seq, vae_keys)
            recon_loss = jnp.mean((raw_dense_seq - recon_seq) ** 2, axis=(0, 2))
            kl_div = -0.5 * jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=-1)
            ood_score = recon_loss + kl_div

        # 1. Pass RAW features as scan inputs — NOT pre-embedded
        # Dense: stack raw features (T, N, sum_F_dense) — much smaller than (T, N, H)
        raw_dense = jnp.concatenate(
            [data.dynamic[name] for name in self.dense_sources], axis=-1
        )  # (T, N, sum_F_dense)

        # Sparse: pass raw features directly (T, N, F_s) per source
        raw_sparse = {name: data.dynamic[name] for name in self.source_registry.sources}

        # 2. Initialize state
        initial_h = jnp.zeros((num_locations, self.hidden_size))
        initial_c = jnp.zeros((num_locations, self.hidden_size))
        initial_staleness = jnp.full((num_sources, num_locations), 1e6)
        initial_state = (initial_h, initial_c, initial_staleness)

        # 3. Scan
        @jax.checkpoint
        def process_one_timestep(
            state_prev, scan_slice, static_data, graph_edges, node_mask, edge_mask
        ):
            h_prev, c_prev, staleness_prev = state_prev
            raw_dense_t, raw_sparse_t, step_key = scan_slice
            drop_key, cell_key = jrandom.split(step_key)

            # --- Embed INSIDE the scan body (recomputed on backward) ---

            # Dense embedding
            dense_emb_list = []
            for name_idx, name in enumerate(self.dense_sources):
                # Slice this source's features from the concatenated raw input
                emb = jax.vmap(self.dense_embedders[name])(
                    jax.lax.dynamic_slice_in_dim(
                        raw_dense_t,
                        self._dense_offsets[name_idx],
                        self._dense_sizes[name_idx],
                        axis=-1,
                    )
                )
                dense_emb_list.append(emb)
            dense_input = jnp.sum(jnp.stack(dense_emb_list), axis=0)
            dense_input = jax.vmap(self.fusion_norm)(dense_input)

            # Sparse embedding (H_s operators)
            source_names = self.source_registry.sources
            obs_embeddings = {}
            obs_valid = {}
            obs_staleness_dict = {}
            new_staleness = staleness_prev

            for idx, name in enumerate(source_names):
                features_t = raw_sparse_t[name]  # (N, F_s)
                emb_t, mask_t = self.source_registry.embed(name, features_t, static_data)

                source_staleness_new = jnp.where(mask_t, 0.0, staleness_prev[idx] + 1.0)
                new_staleness = new_staleness.at[idx].set(source_staleness_new)

                obs_embeddings[name] = emb_t
                obs_valid[name] = mask_t
                obs_staleness_dict[name] = staleness_prev[idx]

            # --- Cell forward ---
            (h_new, c_new), cell_trace = self.spat_temp_lstm(
                dense_input,
                (h_prev, c_prev),
                obs_embeddings,
                obs_valid,
                obs_staleness_dict,
                static_data,
                graph_edges,
                node_mask,
                edge_mask,
                key=cell_key,
            )

            new_state = (h_new, c_new, new_staleness)
            return new_state, h_new

        partial_timestep = partial(
            process_one_timestep,
            static_data=data.static,
            graph_edges=graph_edges,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )

        scan_keys = jrandom.split(scan_key, self.seq_length)
        scan_inputs = (raw_dense, raw_sparse, scan_keys)

        final_state, all_h = jax.lax.scan(partial_timestep, initial_state, scan_inputs)
        final_h, final_c, final_staleness = final_state

        # 4. Predictions
        if self.seq2seq:
            all_h = self.dropout(all_h, key=drop_key2)
            predictions = {
                target: jax.vmap(jax.vmap(head))(all_h) for target, head in self.head.items()
            }
        else:
            final_h = self.dropout(final_h, key=drop_key2)
            predictions = {target: jax.vmap(head)(final_h) for target, head in self.head.items()}

        if self.sequence_vae is not None:
            predictions["vae_ood_score"] = ood_score

        return predictions