from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from data import GraphBatch

from .base_model import BaseModel
from .layers.graph_attn_c import SpatioTemporalLSTMCell
from .layers.static_mlp import StaticMLP


class ST_GATransformer_C(BaseModel):
    hidden_size: int
    seq_length: int
    dense_sources: list[str]
    sparse_sources: list[str]
    seq2seq: bool
    return_weights: bool

    static_embedder: eqx.nn.MLP
    dense_embedder: eqx.nn.MLP
    sparse_embedders: dict[str, StaticMLP]
    sparse_gain_nets: dict[str, eqx.nn.MLP]
    obs_projections: dict[str, eqx.nn.Linear]
    pre_assim_norm: eqx.nn.LayerNorm
    c_to_h: eqx.nn.Linear
    spat_temp_lstm: SpatioTemporalLSTMCell

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
        return_weights: bool,
    ):
        key = jrandom.PRNGKey(seed)
        keys = list(jrandom.split(key, 12))

        # Initializes linear head and target list
        super().__init__(hidden_size, target, head, key=keys.pop(0))

        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.dense_sources = list(dense_sizes.keys())
        self.sparse_sources = list(sparse_sizes.keys())
        self.seq2seq = seq2seq
        self.return_weights = return_weights

        # Embedding network for static features
        self.static_embedder = eqx.nn.MLP(
            in_size=static_size,
            out_size=hidden_size,
            width_size=2 * hidden_size,
            depth=2,
            key=keys.pop(0),
        )

        # Embedding network for concatenated dense dynamic features
        total_dense_size = sum(list(dense_sizes.values()))
        self.dense_embedder = eqx.nn.MLP(
            in_size=total_dense_size,
            out_size=hidden_size,
            width_size=2 * hidden_size,
            depth=2,
            key=keys.pop(0),
        )

        # Embedding and assimilation networks for the sparse sources
        self.sparse_embedders = {}
        self.sparse_gain_nets = {}
        self.obs_projections = {}
        sparse_keys = jrandom.split(keys.pop(0), len(self.sparse_sources))
        for (name, in_size), k in zip(sparse_sizes.items(), sparse_keys):
            self.add_assimilator(name, in_size, assim_sizes.get(name, {}), key=k)

        # Layer norm applied to c before computing innovation
        self.pre_assim_norm = eqx.nn.LayerNorm(hidden_size)

        # Output projection: derives h from c via learned linear + tanh
        self.c_to_h = eqx.nn.Linear(hidden_size, hidden_size, key=keys.pop(0))

        # The combined graph attention LSTM cell
        self.spat_temp_lstm = SpatioTemporalLSTMCell(
            lstm_input_size=hidden_size,
            static_feature_size=hidden_size,
            hidden_size=hidden_size,
            k_hops=k_hops,
            num_heads=num_heads,
            dropout_p=dropout,
            key=keys.pop(0),
        )

    def get_backbone(self) -> list[str]:
        return [
            "static_embedder",
            "dense_embedder",
            "spat_temp_lstm",
            "c_to_h",
            "pre_assim_norm",
            "head",
        ]

    def add_assimilator(
        self, name: str, n_features: int, size_args: dict = {}, *, key: PRNGKeyArray = None
    ):
        if name not in self.sparse_sources:
            self.sparse_sources.append(name)

        key = key if key is not None else jax.random.PRNGKey(0)
        k1, k2, k3 = jrandom.split(key, 3)

        embed_width = size_args.get("embed_width", self.hidden_size)
        embed_depth = size_args.get("embed_depth", 2)
        self.sparse_embedders[name] = StaticMLP(
            dynamic_in_size=n_features,
            static_in_size=self.hidden_size,
            out_size=self.hidden_size,
            width_size=embed_width,
            depth=embed_depth,
            key=k1,
        )

        # Projection from observation embedding space into c-space
        self.obs_projections[name] = eqx.nn.Linear(
            self.hidden_size, self.hidden_size, key=k3
        )

        gain_width = size_args.get("gain_width", self.hidden_size)
        gain_depth = size_args.get("gain_depth", 2)
        # Gain input: projected observation + innovation (both in c-space)
        gain_in_size = 2 * self.hidden_size
        self.sparse_gain_nets[name] = eqx.nn.MLP(
            in_size=gain_in_size,
            out_size=self.hidden_size,
            width_size=gain_width,
            depth=gain_depth,
            key=k2,
        )

    def derive_h(self, c: Array, node_mask: Array) -> Array:
        """Derive hidden state h from cell state c via learned projection + tanh."""
        h = jnp.tanh(jax.vmap(self.c_to_h)(c))
        return h * node_mask[:, jnp.newaxis]

    def __call__(self, data: GraphBatch, key: PRNGKeyArray) -> Array:
        num_locations = data.static.shape[0]
        node_mask = data.node_mask
        edge_mask = data.edge_mask
        graph_edges = data.graph_edges

        # Data Embedding
        static_emb = jax.vmap(self.static_embedder)(data.static)

        dense_feat = jnp.concat([data.dynamic[name] for name in self.dense_sources], axis=-1)
        dense_emb = jax.vmap(jax.vmap(self.dense_embedder))(dense_feat)

        # Embed all sparse sources across the full time dimension
        obs_emb = {}
        for name in self.sparse_sources:
            features = data.dynamic[name]
            obs_mask = ~jnp.any(jnp.isnan(features), axis=-1, keepdims=True)
            safe_features = jnp.nan_to_num(features)
            embedding_mlp = self.sparse_embedders[name]
            emb_vectors = embedding_mlp(safe_features, static_emb)
            # Shape: (seq_length, num_locations, hidden_size), (seq_length, num_locations, 1)
            obs_emb[name] = (emb_vectors, obs_mask)

        # --- Recurrent Processing Loop ---
        @jax.checkpoint
        def process_one_timestep(carry, scan_slice, static_data, graph_edges, node_mask, edge_mask):
            (h_prev, c_prev) = carry
            dense_input, step_obs_emb, step_key = scan_slice

            # 1. Temporal update: LSTM produces new c from input
            c_new = self.spat_temp_lstm.temporal_step(
                dense_input,
                (h_prev, c_prev),
                node_mask,
            )

            # 2. Assimilate sparse observations into c (before spatial propagation)
            if step_obs_emb is not None:
                c_new, assim_trace = self.assimilate_to_cell(c_new, step_obs_emb, node_mask)
            else:
                assim_trace = {}

            # 3. Spatial propagation: GAT spreads corrected c across the graph
            c_spatial, spatial_trace = self.spat_temp_lstm.spatial_step(
                c_new,
                static_data,
                graph_edges,
                node_mask,
                edge_mask,
                key=step_key,
            )

            # 4. Derive h from spatially-propagated c
            h_new = self.derive_h(c_spatial, node_mask)

            new_carry = (h_new, c_spatial)
            weights = {"spatial": spatial_trace, "assim": assim_trace}
            accumulated_outputs = (h_new, weights)

            return new_carry, accumulated_outputs

        # Initial states
        initial_carry = (
            jnp.zeros((num_locations, self.hidden_size)),  # h_0
            jnp.zeros((num_locations, self.hidden_size)),  # c_0
        )

        partial_timestep = partial(
            process_one_timestep,
            static_data=static_emb,
            graph_edges=graph_edges,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )

        scan_keys = jrandom.split(key, self.seq_length)
        scan_inputs = (dense_emb, obs_emb if self.sparse_sources else None, scan_keys)
        final_carry, accumulated = jax.lax.scan(partial_timestep, initial_carry, scan_inputs)

        (final_h, final_c) = final_carry
        all_h, weights = accumulated

        if self.seq2seq:
            predictions = {
                target: jax.vmap(jax.vmap(head))(all_h) for target, head in self.head.items()
            }
        else:
            predictions = {target: jax.vmap(head)(final_h) for target, head in self.head.items()}

        if self.return_weights:
            return predictions, weights
        else:
            return predictions

    def assimilate_to_cell(
        self,
        c: Array,
        step_obs_emb: dict[str, tuple[Array, Array]],
        node_mask: Array,
    ) -> tuple[Array, dict]:
        """
        Assimilates sparse observations into the cell state c.

        Applied BEFORE spatial propagation so that the GAT can spread
        observation-corrected information to neighboring nodes within
        the same timestep.

        For each sparse source:
          1. Projects the observation embedding into c-space.
          2. Computes the innovation (residual) against layer-normed c.
          3. Computes a sigmoid-bounded gain.
          4. Applies the correction only where a fresh observation exists.
        """
        tracing_weights = {}

        for name in self.sparse_sources:
            update_emb_t, mask_t = step_obs_emb[name]  # (num_loc, hidden), (num_loc, 1)

            # Project observation embedding into c-space
            obs_in_c = jax.vmap(self.obs_projections[name])(update_emb_t)

            # Layer norm on c before computing innovation (stabilizes scale)
            c_normed = jax.vmap(self.pre_assim_norm)(c)

            # Innovation in normalized space
            nu = obs_in_c - c_normed

            # Gain computation with sigmoid activation (bounded [0, 1])
            gain_input = jnp.concatenate([obs_in_c, nu], axis=-1)
            gain = jax.nn.sigmoid(jax.vmap(self.sparse_gain_nets[name])(gain_input))

            # Only apply correction where a fresh observation exists
            masked_gain = jnp.where(mask_t, gain, 0.0)

            # Residual update to raw c (not normed c)
            c = c + masked_gain * nu

            tracing_weights[name] = {"gain": masked_gain, "nu": nu}

        return c, tracing_weights