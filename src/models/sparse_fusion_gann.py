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
# from .layers.graph_attn import SpatioTemporalLSTMCell


def segment_softmax(logits: Array, segment_ids: Array, num_segments: int) -> Array:
    """Compute softmax over segments safely."""
    # Subtract max for numerical stability
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

    # Projections for Q, K, V
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
        h: Array,
        x_s: Array,
        edge_index: tuple[Array, Array],
        node_mask: Array,
        edge_mask: Array,
        *,
        key: PRNGKeyArray = None,
    ) -> tuple[Array, Array]:
        num_nodes = h.shape[0]
        src, dest = edge_index  # src=upstream, dest=downstream

        # 1. Condition hidden state with static features
        static_emb = jax.vmap(self.static_proj)(x_s)
        h_cond = jax.vmap(self.norm)(h + static_emb)

        # 2. Project to Q, K, V
        # Reshape to (N, Heads, Dim)
        Q = jax.vmap(self.query_proj)(h_cond).reshape(num_nodes, self.num_heads, self.head_dim)
        K = jax.vmap(self.key_proj)(h_cond).reshape(num_nodes, self.num_heads, self.head_dim)
        V = jax.vmap(self.value_proj)(h_cond).reshape(num_nodes, self.num_heads, self.head_dim)

        # 3. Sparse Attention: Compute scores only on edges
        # Q comes from DEST (receiver), K comes from SRC (sender)
        Q_dest = Q[dest]  # (E, Heads, Dim)
        K_src = K[src]  # (E, Heads, Dim)
        V_src = V[src]  # (E, Heads, Dim)

        # Dot product attention: (E, Heads)
        scale = jnp.sqrt(self.head_dim)
        scores = jnp.einsum("ehd,ehd->eh", Q_dest, K_src) / scale

        # Mask edges that shouldn't exist (if edge_mask is used for padding/graph structure)
        scores = jnp.where(edge_mask[:, None], scores, -1e9)

        # 4. Softmax over incoming edges for each destination node
        attn_weights = segment_softmax(scores, dest, num_nodes)  # (E, Heads)

        # 5. Aggregate Values
        # Weight values by attention
        weighted_V = V_src * attn_weights[:, :, None]  # (E, Heads, Dim)

        # Sum messages at destination
        # Flatten heads for scatter sum: (E, Hidden)
        weighted_V_flat = weighted_V.reshape(weighted_V.shape[0], self.hidden_size)

        # Aggregate
        aggr_out = jax.ops.segment_sum(weighted_V_flat, dest, num_nodes)

        # 6. Final projection
        output = jax.vmap(self.out_proj)(aggr_out)

        # Apply node mask
        output = output * node_mask[:, None]

        # Return sparse weights (E, Heads) instead of dense matrix (N, N)
        return output, attn_weights


class UpstreamPropagation(eqx.Module):
    """
    Propagate information upstream (against physical flow direction).

    Information flows: outlet -> headwaters
    Each node receives from exactly one downstream neighbor (if any).
    Simple gated propagation, no attention needed.
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
        h: Array,
        x_s: Array,
        edge_index: tuple[Array, Array],
        node_mask: Array,
        edge_mask: Array,
    ) -> tuple[Array, Array]:
        """
        Each node receives from its single downstream neighbor.

        Edge convention: (src, dest) means src -> dest physically.
        For upstream propagation: src receives from dest (reversed).
        """
        num_nodes = h.shape[0]
        src, dest = edge_index

        # Reverse direction: dest sends to src
        sender = dest  # downstream node
        receiver = src  # upstream node

        sender_input = jnp.concatenate([h[sender], x_s[sender]], axis=-1)
        message = jax.vmap(self.message_net)(sender_input)

        gate_input = jnp.concatenate([h[sender], h[receiver], x_s[sender], x_s[receiver]], axis=-1)
        gate = jax.nn.sigmoid(jax.vmap(self.gate_net)(gate_input))

        gated_message = message * gate * edge_mask[:, jnp.newaxis]

        # Each upstream node receives from exactly one downstream node
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
        h: Array,
        x_s: Array,
        edge_index: tuple[Array, Array],
        node_mask: Array,
        edge_mask: Array,
        *,
        key: PRNGKeyArray = None,
    ) -> tuple[Array, dict]:
        h_norm = jax.vmap(self.norm)(h)

        # Downstream: aggregate from tributaries
        downstream_msg, downstream_weight = self.downstream_agg(
            h_norm, x_s, edge_index, node_mask, edge_mask, key=key
        )

        # Upstream: receive from downstream neighbor
        upstream_msg, upstream_gate = self.upstream_prop(
            h_norm, x_s, edge_index, node_mask, edge_mask
        )

        combined = jnp.concatenate([downstream_msg, upstream_msg], axis=-1)
        message = jax.vmap(self.combine_proj)(combined)

        output = h + message

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


class SpatioTemporalLSTMCell(eqx.Module):
    """
    Recurrent cell: LSTM temporal + Graph spatial.
    """

    gat: StackedGraphAttention
    lstm_cell: eqx.nn.LSTMCell

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

        self.gat = StackedGraphAttention(
            static_feature_size=static_feature_size,
            hidden_size=hidden_size,
            k_hops=k_hops,
            num_heads=num_heads,
            dropout_p=dropout_p,
            key=gat_key,
        )

        self.lstm_cell = eqx.nn.LSTMCell(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            key=lstm_key,
        )

    def __call__(
        self,
        x_t: Array,
        state: tuple[Array, Array],
        node_features: Array,
        edge_index: tuple[Array, Array],
        node_mask: Array,
        edge_mask: Array,
        *,
        key: PRNGKeyArray,
    ) -> tuple[tuple[Array, Array], tuple]:
        h_prev, c_prev = state
        k1, k2 = jax.random.split(key)

        h_temporal, c_new = jax.vmap(self.lstm_cell)(x_t, (h_prev, c_prev))
        h_temporal = h_temporal * node_mask[:, jnp.newaxis]
        c_new = c_new * node_mask[:, jnp.newaxis]

        h_new, spatial_trace, z, r = self.gat(
            h_temporal,
            node_features,
            edge_index=edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
            key=k1,
        )

        new_state = (h_new, c_new)
        spatial_trace["z_gate"] = z
        spatial_trace["r_gate"] = r

        return new_state, spatial_trace


class ObservationFusionModule(eqx.Module):
    """Source-agnostic observation fusion using cross-attention."""

    hidden_size: int
    null_obs_emb: Array
    obs_query_proj: eqx.nn.Linear
    obs_key_proj: eqx.nn.Linear
    obs_value_proj: eqx.nn.Linear
    staleness_embedding: eqx.nn.MLP
    output_gate: eqx.nn.Linear
    norm: eqx.nn.LayerNorm

    def __init__(self, hidden_size: int, *, key: PRNGKeyArray):
        keys = jax.random.split(key, 6)
        self.hidden_size = hidden_size

        self.null_obs_emb = jax.random.normal(keys[0], (hidden_size,))
        self.obs_query_proj = eqx.nn.Linear(hidden_size, hidden_size, key=keys[1])
        self.obs_key_proj = eqx.nn.Linear(hidden_size, hidden_size, key=keys[2])
        self.obs_value_proj = eqx.nn.Linear(hidden_size, hidden_size, key=keys[3])

        self.staleness_embedding = eqx.nn.MLP(
            in_size=1,
            out_size=hidden_size,
            width_size=hidden_size // 2,
            depth=2,
            key=keys[4],
        )

        self.output_gate = eqx.nn.Linear(hidden_size * 2, hidden_size, key=keys[5])
        # A bias of +2.0 makes sigmoid(x) ≈ 0.88.
        # This forces the model to use the fusion update early in training.
        new_bias = self.output_gate.bias + 2.0
        self.output_gate = eqx.tree_at(lambda g: g.bias, self.output_gate, new_bias)

        self.norm = eqx.nn.LayerNorm(hidden_size)

    def __call__(
        self,
        h: Array,
        obs_embeddings: list[Array],
        obs_staleness: list[Array],
        obs_valid: list[Array],
    ) -> tuple[Array, dict]:
        num_sources = len(obs_embeddings)

        if num_sources == 0:
            return h, {"weights": None, "valid_obs": None, "gate": None}

        # Real observational data
        obs_stack = jnp.stack(obs_embeddings, axis=1)
        staleness_stack = jnp.stack(obs_staleness, axis=1)
        valid_stack = jnp.stack(obs_valid, axis=1)

        log_staleness = jnp.log1p(staleness_stack[:, :, jnp.newaxis])
        staleness_emb = jax.vmap(jax.vmap(self.staleness_embedding))(log_staleness)
        modulated_obs = obs_stack + staleness_emb

        # Null token
        batch_size = h.shape[0]
        null_emb = jnp.tile(self.null_obs_emb[None, None, :], (batch_size, 1, 1))
        null_valid = jnp.ones((batch_size, 1), dtype=jnp.bool_)  # null is always valid

        # concat obs with null
        full_obs = jnp.concatenate([modulated_obs, null_emb], axis=1)
        full_valid = jnp.concatenate([valid_stack, null_valid], axis=1)

        # sparse attention
        query = jax.vmap(self.obs_query_proj)(h)
        keys = jax.vmap(jax.vmap(self.obs_key_proj))(full_obs)
        values = jax.vmap(jax.vmap(self.obs_value_proj))(full_obs)

        scale = jnp.sqrt(self.hidden_size)
        scores = jnp.einsum("lh,lsh->ls", query, keys) / scale

        # Mask invalid, leaving Null as the fallback
        scores = jnp.where(full_valid, scores, -1e9)

        # Never collapses to -inf because of the null obs
        weights = jax.nn.softmax(scores, axis=-1)

        fused_obs = jnp.einsum("ls,lsh->lh", weights, values)

        gate_input = jnp.concatenate([h, fused_obs], axis=-1)
        gate = jax.nn.sigmoid(jax.vmap(self.output_gate)(gate_input))

        # Only apply gate if we actually had REAL data (optional, but good for stability)
        # If we only attended to Null, we effectively want update=0.
        # But letting the model learn that is safer than hard-coding it.
        any_real_valid = jnp.any(valid_stack, axis=-1, keepdims=True)
        effective_gate = jnp.where(any_real_valid, gate, 0.0)

        # residual update
        update = jax.vmap(self.norm)(fused_obs)
        h_new = h + (effective_gate * update)

        return h_new, {"weights": weights, "valid_obs": full_valid, "gate": effective_gate}


class SourceEmbedderRegistry(eqx.Module):
    """Registry of source-specific embedders mapping to a common hidden space."""

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

        # Single vmap over spatial dimension N
        normed_emb = jax.vmap(self.norms[name])(raw_emb)

        # Expand 1D mask (N,) to match (N, H)
        masked_emb = jnp.where(obs_mask[:, jnp.newaxis], normed_emb, 0.0)

        return masked_emb, obs_mask

    @property
    def sources(self) -> list[str]:
        return list(self.embedders.keys())


class SparseFusionGRNN(BaseModel):
    """
    Spatio-Temporal Graph Recurrent Neural Network with:
    - Bidirectional graph message passing (upstream attention + downstream/backward gating)
    - Source-agnostic observation fusion via cross-attention
    - Modular source embedders for easy addition of new observation types
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
    obs_fusion: ObservationFusionModule
    spat_temp_lstm: SpatioTemporalLSTMCell
    dropout: eqx.nn.Dropout

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
            # raise NotImplementedError()
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
        self.fusion_norm = eqx.nn.LayerNorm(hidden_size)

        # Sparse observation handling
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

        self.obs_fusion = ObservationFusionModule(hidden_size, key=keys.pop(0))

        # Spatio-temporal cell
        self.spat_temp_lstm = SpatioTemporalLSTMCell(
            lstm_input_size=hidden_size,
            static_feature_size=static_size,
            hidden_size=hidden_size,
            k_hops=k_hops,
            num_heads=num_heads,
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
        """Add a new observation source."""
        key = key if key is not None else jax.random.PRNGKey(0)
        self.source_registry.register_source(
            name,
            n_features,
            embed_width=size_args.get("embed_width"),
            embed_depth=size_args.get("embed_depth", 2),
            key=key,
        )

    @property
    def sparse_sources(self) -> list[str]:
        return self.source_registry.sources

    def __call__(self, data: GraphBatch, key: PRNGKeyArray) -> Array:
        num_locations = data.static.shape[0]
        node_mask = data.node_mask
        edge_mask = data.edge_mask
        graph_edges = data.graph_edges

        vae_key, drop_key1, drop_key2, scan_key = jrandom.split(key, 4)

        # 0 VAE out of distribution score
        if self.sequence_vae is not None:
            raw_dense_list = [data.dynamic[name] for name in self.dense_sources]
            raw_dense_seq = jnp.concatenate(raw_dense_list, axis=-1)

            vae_keys = jrandom.split(vae_key, num_locations)

            # out_axes=(1, 0, 0) maps recon_seq to (S, N, F) and mu/logvar to (N, L)
            recon_seq, mu, logvar = jax.vmap(self.sequence_vae, in_axes=(1, 0), out_axes=(1, 0, 0))(
                raw_dense_seq, vae_keys
            )

            # Average over seq_length (0) and num_features (2) to isolate num_locations (1)
            recon_loss = jnp.mean((raw_dense_seq - recon_seq) ** 2, axis=(0, 2))

            # Sum over latent_dim (-1) to isolate num_locations (0)
            kl_div = -0.5 * jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=-1)

            # Final ood_score shape: (num_locations,)
            ood_score = recon_loss + kl_div

        # 1. Dense embedding
        dense_emb_list = []
        for name in self.dense_sources:
            features = data.dynamic[name]
            emb_mlp = self.dense_embedders[name]
            dense_emb_list.append(jax.vmap(jax.vmap(emb_mlp))(features))
        dense_emb = jnp.sum(jnp.stack(dense_emb_list), axis=0)
        dense_emb_norm = jax.vmap(jax.vmap(self.fusion_norm))(dense_emb)
        dense_emb_norm = self.dropout(dense_emb_norm, key=drop_key1)

        # 2. Sparse observation embedding (Time-distributed)
        obs_embs_time = {}
        for name in self.source_registry.sources:
            features = data.dynamic[name]
            # Vmap over the time dimension (axis 0)
            emb, mask = jax.vmap(self.source_registry.embed, in_axes=(None, 0, None))(
                name, features, data.static
            )
            obs_embs_time[name] = (emb, mask)

        # 3. Initialize recurrent state and global memory carry
        init_global_memory = (
            jnp.zeros((num_locations, self.hidden_size)),  # pooled_emb
            jnp.full((num_locations,), 1e6),  # staleness
            jnp.zeros((num_locations,), dtype=jnp.bool_),  # seen_mask
        )

        initial_h = jnp.zeros((num_locations, self.hidden_size))
        initial_c = jnp.zeros((num_locations, self.hidden_size))
        initial_state = (initial_h, initial_c, init_global_memory)

        # 4. Recurrent loop
        @jax.checkpoint
        def process_one_timestep(
            state_prev, scan_slice, static_data, graph_edges, node_mask, edge_mask
        ):
            h_prev, c_prev, memory_prev = state_prev
            prev_emb, prev_stale, prev_seen = memory_prev
            dense_input, step_obs, step_key = scan_slice

            # LSTM step
            (h_new, c_new), spatial_trace = self.spat_temp_lstm(
                dense_input,
                (h_prev, c_prev),
                static_data,
                graph_edges,
                node_mask,
                edge_mask,
                key=step_key,
            )

            # Observation pooling and assim
            if not step_obs:
                memory_new = memory_prev
                fusion_trace = {"gate": None, "valid_obs": None, "weights": None}
            else:
                valid_embs = []
                masks = []
                for name, (emb_t, mask_t) in step_obs.items():
                    valid_embs.append(jnp.where(mask_t[:, jnp.newaxis], emb_t, 0.0))
                    masks.append(mask_t)

                stacked_embs = jnp.stack(valid_embs)  # (S, N, H)
                stacked_masks = jnp.stack(masks)  # (S, N)

                # Sum across sources
                sum_emb = jnp.sum(stacked_embs, axis=0)  # (N, H)

                # Dynamic denominator for mean pooling
                valid_count = jnp.sum(stacked_masks, axis=0)[:, jnp.newaxis]  # (N, 1)

                # Mean pool, protecting against division by zero
                pooled_step_emb = sum_emb / jnp.maximum(valid_count, 1.0)
                any_valid_mask = jnp.any(stacked_masks, axis=0)

                # 2. Update the SINGLE global memory state
                new_emb = jnp.where(any_valid_mask[:, jnp.newaxis], pooled_step_emb, prev_emb)
                new_stale = jnp.where(any_valid_mask, 0.0, prev_stale + 1.0)
                new_seen = prev_seen | any_valid_mask

                memory_new = (new_emb, new_stale, new_seen)
                h_new, fusion_trace = self._fuse_global_memory(h_new, memory_new)

            new_state = (h_new, c_new, memory_new)
            accumulated_outputs = (h_new, spatial_trace, fusion_trace)
            return new_state, accumulated_outputs

        partial_timestep = partial(
            process_one_timestep,
            static_data=data.static,
            graph_edges=graph_edges,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )

        scan_keys = jrandom.split(scan_key, self.seq_length)
        scan_inputs = (dense_emb_norm, obs_embs_time, scan_keys)
        final_state, accumulated = jax.lax.scan(partial_timestep, initial_state, scan_inputs)

        final_h, final_c, final_memory = final_state
        all_h, spatial_trace, fusion_trace = accumulated

        # 5. Predictions (identical to original)
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

        if self.supervised_attn:
            predictions["attention_weights"] = fusion_trace.get("weights", None)
            predictions["valid_obs"] = fusion_trace.get("valid_obs", None)

        if self.return_weights:
            return predictions, {"spatial": spatial_trace, "fusion": fusion_trace}
        return predictions

    def _fuse_global_memory(
        self,
        h: Array,
        global_memory: tuple[Array, Array, Array],
    ) -> tuple[Array, dict]:
        emb, staleness, has_been_seen = global_memory

        if self.use_obs_memory:
            valid = has_been_seen & (staleness < self.staleness_threshold)
        else:
            valid = staleness == 0

        # Modulate the pooled embedding with temporal decay
        log_staleness = jnp.log1p(staleness[:, jnp.newaxis])

        # The original staleness_embedding expects a feature dim of 1
        staleness_emb = jax.vmap(self.obs_fusion.staleness_embedding)(log_staleness)
        modulated_obs = emb + staleness_emb

        # Compute fusion gate directly against the hidden state
        gate_input = jnp.concatenate([h, modulated_obs], axis=-1)
        gate = jax.nn.sigmoid(jax.vmap(self.obs_fusion.output_gate)(gate_input))

        # Suppress update if the global memory is stale or empty
        effective_gate = jnp.where(valid[:, jnp.newaxis], gate, 0.0)

        # Residual update
        update = jax.vmap(self.obs_fusion.norm)(modulated_obs)
        h_new = h + (effective_gate * update)

        fusion_info = {
            "gate": effective_gate,
            "valid_obs": valid[:, jnp.newaxis],
        }

        return h_new, fusion_info
