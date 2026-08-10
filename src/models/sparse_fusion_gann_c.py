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


# ============================================================
# Topology Preprocessing (call once per graph, outside JIT)
# ============================================================


def precompute_tree_topology(src, dest, num_nodes, num_dilations=5):
    """
    Pre-compute tree topology structures for bidirectional message passing.

    Call ONCE per graph topology (outside of JIT/training loop). Returns
    JAX arrays to be stored in the dataset and passed to the model.

    Assumes a directed tree where edges point upstream -> downstream (src -> dest).
    Each non-headwater node has exactly 2 upstream parents.
    Each non-outlet node has exactly 1 downstream child.

    Args:
        src: (E,) source indices (upstream nodes)
        dest: (E,) destination indices (downstream nodes)
        num_nodes: total number of nodes in the graph
        num_dilations: number of dilation levels (upstream reach = 2^num_dilations - 1)

    Returns:
        dict with:
            level_nodes: (num_levels, max_per_level) node indices per level, padded
            level_mask: (num_levels, max_per_level) validity mask
            level_parents: (num_levels, max_per_level, 2) upstream parent indices
            level_parent_mask: (num_levels, max_per_level, 2) parent validity
            dilated_downstream: (num_nodes, num_dilations) node at distance 2^k downstream
            dilated_valid: (num_nodes, num_dilations) dilation validity mask
            num_levels: int
            num_dilations: int
    """
    import numpy as np
    from collections import deque

    src_np = np.asarray(src)
    dest_np = np.asarray(dest)

    # Build adjacency structures
    upstream_parents = [[] for _ in range(num_nodes)]
    downstream_child = np.full(num_nodes, -1, dtype=np.int32)

    for s, d in zip(src_np, dest_np):
        upstream_parents[int(d)].append(int(s))
        downstream_child[int(s)] = int(d)

    # Compute topological levels via BFS from headwaters
    levels = np.full(num_nodes, -1, dtype=np.int32)

    queue = deque()
    for i in range(num_nodes):
        if len(upstream_parents[i]) == 0:
            levels[i] = 0
            queue.append(i)

    while queue:
        node = queue.popleft()
        child = downstream_child[node]
        if 0 <= child < num_nodes:
            parent_levels = [levels[p] for p in upstream_parents[child] if levels[p] >= 0]
            if len(parent_levels) == len(upstream_parents[child]):
                levels[child] = max(parent_levels) + 1
                queue.append(child)

    # Handle unreachable nodes (assign level 0)
    levels[levels < 0] = 0
    num_levels = int(levels.max()) + 1

    # Group nodes by level
    level_groups = [[] for _ in range(num_levels)]
    for i in range(num_nodes):
        level_groups[int(levels[i])].append(i)

    max_per_level = max(len(g) for g in level_groups) if level_groups else 1

    # Build padded level arrays
    level_nodes_arr = np.zeros((num_levels, max_per_level), dtype=np.int32)
    level_mask_arr = np.zeros((num_levels, max_per_level), dtype=np.bool_)
    level_parents_arr = np.zeros((num_levels, max_per_level, 2), dtype=np.int32)
    level_parent_mask_arr = np.zeros((num_levels, max_per_level, 2), dtype=np.bool_)

    for l in range(num_levels):
        for idx, node in enumerate(level_groups[l]):
            level_nodes_arr[l, idx] = node
            level_mask_arr[l, idx] = True

            parents = upstream_parents[node]
            for p_idx, parent in enumerate(parents[:2]):
                level_parents_arr[l, idx, p_idx] = parent
                level_parent_mask_arr[l, idx, p_idx] = True
            # Pad unused parent slots with self-reference (will be masked)
            for p_idx in range(len(parents[:2]), 2):
                level_parents_arr[l, idx, p_idx] = node

    # Compute dilated downstream indices for upstream propagation
    dilated_downstream_arr = np.zeros((num_nodes, num_dilations), dtype=np.int32)
    dilated_valid_arr = np.zeros((num_nodes, num_dilations), dtype=np.bool_)

    for k in range(num_dilations):
        distance = 2**k
        for i in range(num_nodes):
            current = i
            valid = True
            for _ in range(distance):
                nxt = downstream_child[current]
                if nxt < 0 or nxt >= num_nodes:
                    valid = False
                    break
                current = nxt
            dilated_downstream_arr[i, k] = current if valid else i
            dilated_valid_arr[i, k] = valid and (current != i)

    return {
        "level_nodes": jnp.array(level_nodes_arr),
        "level_mask": jnp.array(level_mask_arr),
        "level_parents": jnp.array(level_parents_arr),
        "level_parent_mask": jnp.array(level_parent_mask_arr),
        "dilated_downstream": jnp.array(dilated_downstream_arr),
        "dilated_valid": jnp.array(dilated_valid_arr),
        "num_levels": num_levels,
        "num_dilations": num_dilations,
    }


# ============================================================
# Graph Message Passing Modules
# ============================================================


class DownstreamTreeScan(eqx.Module):
    """
    Full downstream propagation from headwaters to outlet via
    level-by-level attention-weighted residual aggregation.

    Each node has a learned routing coefficient τ_i conditioned on static
    features that controls how much upstream information updates its state.
    Because the scan cascades (each level's updated state becomes the parent
    for the next level), attenuation at reservoir nodes naturally reduces
    the signal seen by all downstream nodes.

    τ is NOT state-dependent — it is a structural property of each node
    (like channel conductance), preventing the model from dynamically
    choosing to ignore spatial information.

    Equations at each level for node i with upstream parents j:
        Q_i = W_Q . c_i
        K_j = W_K . c_j
        alpha_{i,j} = softmax(Q_i . K_j / sqrt(d))
        m_down_i = sum_j( alpha_{i,j} . (c_j - c_i) )
        c_i = c_i + τ_i * m_down_i

    Where τ_i = sigmoid(w_route . static_i + b_route) is the routing coefficient.
    """

    query_proj: eqx.nn.Linear
    key_proj: eqx.nn.Linear
    routing_proj: eqx.nn.Linear
    hidden_size: int
    tau_floor: float

    def __init__(self, hidden_size, static_feature_size, tau_floor=0.05, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.hidden_size = hidden_size
        self.tau_floor = tau_floor
        self.query_proj = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=k1)
        self.key_proj = eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=k2)
        self.routing_proj = eqx.nn.Linear(static_feature_size, 1, key=k3)

    def compute_routing_coefficients(self, static_features: Array) -> Array:
        """
        Compute per-node routing coefficients from static features.
        """
        raw = jax.vmap(self.routing_proj)(static_features)
        tau = self.tau_floor + (1.0 - self.tau_floor) * jax.nn.sigmoid(raw)
        return tau

    def __call__(
        self, c, tau, level_nodes, level_mask, level_parents, level_parent_mask, node_mask
    ):
        """
        Args:
            c: (N, H) cell state
            tau: (N, 1) routing coefficients (precomputed from static features)
            level_nodes: (num_levels, max_per_level) node indices per level
            level_mask: (num_levels, max_per_level) validity mask
            level_parents: (num_levels, max_per_level, 2) upstream parent indices
            level_parent_mask: (num_levels, max_per_level, 2) parent validity
            node_mask: (N,) valid node mask

        Returns:
            c_propagated: (N, H) state after full downstream propagation
            attn_weights: (num_levels, max_per_level, 2) attention scores
        """

        def process_level(c, level_slice):
            # level_slice only contains (level_nodes, level_mask)
            nodes, mask = level_slice

            c_nodes = c[nodes]
            
            # Dynamically index the global (total_nodes, 2) arrays
            parents = level_parents[nodes]
            parent_mask = level_parent_mask[nodes]

            c_parents = c[parents.reshape(-1)].reshape(
                parents.shape[0], 2, self.hidden_size
            )

            Q = jax.vmap(self.query_proj)(c_nodes)
            K = jax.vmap(jax.vmap(self.key_proj))(c_parents)

            scale = jnp.sqrt(jnp.float32(self.hidden_size))
            scores = jnp.einsum("mh,mph->mp", Q, K) / scale
            scores = jnp.where(parent_mask, scores, -1e9)
            alpha = jax.nn.softmax(scores, axis=-1)

            diff = c_parents - c_nodes[:, None, :]
            msg = jnp.einsum("mp,mph->mh", alpha, diff)

            tau_nodes = tau[nodes]
            msg = tau_nodes * msg

            has_any_parent = jnp.any(parent_mask, axis=-1)
            msg = msg * mask[:, None] * has_any_parent[:, None]

            c = c.at[nodes].add(msg)
            return c, alpha

        # Isolate sequence-level arrays for JAX scan iteration
        level_data = (level_nodes, level_mask)
        c_out, attn_weights = jax.lax.scan(process_level, c, level_data)
        c_out = c_out * node_mask[:, None]

        return c_out, attn_weights


class UpstreamDilatedPropagation(eqx.Module):
    """
    Upstream propagation via dilated gated messages from progressively
    distant downstream nodes.

    At dilation level k, each node i receives a gated message from the
    node j_k that is 2^k hops downstream:

        g_k = sigma(MLP_k(c_{j_k}, c_i))
        m_k = g_k * (c_{j_k} - c_i)

    Total upstream message: m_up_i = sum_k m_k

    Gates are state-dependent here (unlike downstream routing coefficients)
    because upstream influence (boundary conditions) is genuinely dynamic —
    backwater effects depend on current conditions, not just network structure.

    With num_dilations=5 (distances 1, 2, 4, 8, 16):
        Total reach = 31 hops upstream from outlet.
    """

    gate_nets: list
    hidden_size: int
    num_dilations: int

    def __init__(self, hidden_size, num_dilations, *, key):
        self.hidden_size = hidden_size
        self.num_dilations = num_dilations
        keys = jax.random.split(key, num_dilations)
        self.gate_nets = [
            eqx.nn.MLP(
                in_size=hidden_size * 2,
                out_size=hidden_size,
                width_size=hidden_size,
                depth=1,
                key=keys[k],
            )
            for k in range(num_dilations)
        ]

    def __call__(self, c, dilated_downstream, dilated_valid, node_mask):
        """
        Args:
            c: (N, H) cell state (already downstream-propagated)
            dilated_downstream: (N, num_dilations) downstream node at distance 2^k
            dilated_valid: (N, num_dilations) validity mask
            node_mask: (N,) valid node mask

        Returns:
            upstream_msg: (N, H) total upstream message
            gate_values: (num_dilations, N, H) gate activations
        """
        total_msg = jnp.zeros_like(c)
        gate_values = []

        for k in range(self.num_dilations):
            downstream_idx = dilated_downstream[:, k]
            valid = dilated_valid[:, k]

            c_downstream = c[downstream_idx]

            gate_input = jnp.concatenate([c_downstream, c], axis=-1)
            gate = jax.nn.sigmoid(jax.vmap(self.gate_nets[k])(gate_input))

            diff = c_downstream - c
            msg = gate * diff * valid[:, None]

            total_msg = total_msg + msg
            gate_values.append(gate * valid[:, None])

        total_msg = total_msg * node_mask[:, None]
        gate_stack = jnp.stack(gate_values, axis=0)

        return total_msg, gate_stack


class BidirectionalGraphMessagePassing(eqx.Module):
    """
    Full bidirectional message passing across the river network tree.

    Downstream: Level-by-level cascading scan with routing coefficients.
        - Routing coeff τ is static-feature-conditioned (structural, not dynamic)
        - Reservoir nodes learn low τ, attenuating upstream signals
        - Attenuation cascades naturally through tree structure
        - Cannot be "turned off" globally (τ varies per node based on features)

    Upstream: Dilated gated propagation from distant downstream nodes.
        - Gates ARE state-dependent (boundary conditions are dynamic)
        - Exponential reach via dilation

    Final update: c_t = c^(s) + m_down + m_up
    """

    downstream: DownstreamTreeScan
    upstream: UpstreamDilatedPropagation
    norm: eqx.nn.LayerNorm
    hidden_size: int

    def __init__(self, hidden_size, static_feature_size, num_dilations, *, key):
        k1, k2 = jax.random.split(key)
        self.hidden_size = hidden_size
        self.downstream = DownstreamTreeScan(hidden_size, static_feature_size, key=k1)
        self.upstream = UpstreamDilatedPropagation(hidden_size, num_dilations, key=k2)
        self.norm = eqx.nn.LayerNorm(hidden_size)

    def __call__(self, c, static_features, tree_topology, node_mask):
        """
        Args:
            c: (N, H) assimilated cell state c^(s)
            static_features: (N, static_size) for routing coefficient computation
            tree_topology: dict with precomputed topology arrays
            node_mask: (N,) valid node mask

        Returns:
            c_updated: (N, H) spatially-updated cell state
            trace: dict with attention, gates, and routing coefficients
        """
        # Compute routing coefficients from static features (fixed per node)
        tau = self.downstream.compute_routing_coefficients(static_features)  # (N, 1)

        # Normalize for stable message computation
        c_norm = jax.vmap(self.norm)(c)

        # 1. Downstream: cascading tree scan with routing attenuation
        c_after_down, down_attn = self.downstream(
            c_norm,
            tau,
            tree_topology["level_nodes"],
            tree_topology["level_mask"],
            tree_topology["level_parents"],
            tree_topology["level_parent_mask"],
            node_mask,
        )
        m_down = c_after_down - c_norm

        # 2. Upstream: dilated from downstream-propagated state
        m_up, up_gates = self.upstream(
            c_after_down,
            tree_topology["dilated_downstream"],
            tree_topology["dilated_valid"],
            node_mask,
        )

        # 3. Direct residual update
        c_updated = c + m_down + m_up

        trace = {
            "downstream_attention": down_attn,
            "upstream_gates": up_gates,
            "routing_coefficients": tau,
        }

        return c_updated, trace


# ============================================================
# Sequential Kalman Assimilation
# ============================================================


class SequentialKalmanAssimilation(eqx.Module):
    """
    Sequential Kalman-style data assimilation on the LSTM cell state.

    K_{s,t} = sigma(W_{s,k} . [c_t^{s-1}, z'_{s,t}, log(1+dt_{s,t})] + b_{s,k})
    c_t^{s} = c_t^{s-1} + K_{s,t} * (z'_{s,t} - c_t^{s-1})

    Gain depends on the CURRENT evolving cell state (c_t^{s-1}).
    """

    hidden_size: int
    source_names: list[str]
    gain_networks: dict[str, eqx.nn.Linear]

    def __init__(self, hidden_size: int, source_names: list[str], *, key: PRNGKeyArray):
        self.hidden_size = hidden_size
        self.source_names = list(source_names)
        self.gain_networks = {}

        gain_input_size = hidden_size * 2 + 1
        keys = jax.random.split(key, max(len(source_names), 1))
        for i, name in enumerate(source_names):
            self.gain_networks[name] = eqx.nn.Linear(gain_input_size, hidden_size, key=keys[i])

    def register_source(self, name: str, *, key: PRNGKeyArray):
        gain_input_size = self.hidden_size * 2 + 1
        self.gain_networks[name] = eqx.nn.Linear(gain_input_size, self.hidden_size, key=key)
        self.source_names.append(name)

    def __call__(
        self,
        c: Array,
        obs_embeddings: dict[str, Array],
        obs_valid: dict[str, Array],
        obs_staleness: dict[str, Array],
    ) -> tuple[Array, dict]:
        gains_list = []

        for name in self.source_names:
            if name not in obs_embeddings:
                gains_list.append(jnp.zeros_like(c))
                continue

            z_prime = obs_embeddings[name]
            valid = obs_valid[name]
            staleness = obs_staleness[name]

            log_dt = jnp.log1p(staleness)[:, None]
            gain_input = jnp.concatenate([c, z_prime, log_dt], axis=-1)
            K = jax.nn.sigmoid(jax.vmap(self.gain_networks[name])(gain_input))

            innovation = z_prime - c
            update = K * innovation
            c = c + jnp.where(valid[:, None], update, 0.0)

            gains_list.append(K * valid[:, None])

        kalman_gains = (
            jnp.stack(gains_list) if gains_list else jnp.zeros((0, c.shape[0], self.hidden_size))
        )

        return c, {"kalman_gains": kalman_gains}


# ============================================================
# Spatio-Temporal LSTM Cell
# ============================================================


class SpatioTemporalLSTMCell(eqx.Module):
    """
    Modified LSTM cell with:
        1. LSTM Forecast: c_t' = f*c_{t-1} + i*c_tilde
        2. Sequential Kalman Assimilation on c_t'
        3. Bidirectional Graph Message Passing on assimilated c
        4. LSTM Output: h_t = o_t * tanh(c_final)
    """

    graph_mp: BidirectionalGraphMessagePassing
    kalman: SequentialKalmanAssimilation
    hidden_size: int
    input_size: int

    weight_ih: Array
    weight_hh: Array
    bias: Array

    def __init__(
        self,
        lstm_input_size: int,
        hidden_size: int,
        static_feature_size: int,
        num_dilations: int,
        source_names: list[str],
        *,
        key: PRNGKeyArray,
    ):
        graph_key, kalman_key, lstm_key = jax.random.split(key, 3)

        self.hidden_size = hidden_size
        self.input_size = lstm_input_size

        self.graph_mp = BidirectionalGraphMessagePassing(
            hidden_size=hidden_size,
            static_feature_size=static_feature_size,
            num_dilations=num_dilations,
            key=graph_key,
        )

        self.kalman = SequentialKalmanAssimilation(
            hidden_size=hidden_size,
            source_names=source_names,
            key=kalman_key,
        )

        k1, k2 = jax.random.split(lstm_key)
        scale_ih = 1.0 / jnp.sqrt(lstm_input_size)
        scale_hh = 1.0 / jnp.sqrt(hidden_size)
        self.weight_ih = jax.random.uniform(
            k1, (4 * hidden_size, lstm_input_size), minval=-scale_ih, maxval=scale_ih
        )
        self.weight_hh = jax.random.uniform(
            k2, (4 * hidden_size, hidden_size), minval=-scale_hh, maxval=scale_hh
        )
        bias = jnp.zeros(4 * hidden_size)
        self.bias = bias.at[hidden_size : 2 * hidden_size].set(1.0)

    def _lstm_forecast(self, x: Array, h: Array, c: Array) -> tuple[Array, Array]:
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
        static_features: Array,
        tree_topology: dict,
        node_mask: Array,
        *,
        key: PRNGKeyArray,
    ) -> tuple[tuple[Array, Array], dict]:
        h_prev, c_prev = state

        # 1. LSTM Forecast
        c_prime, o_t = jax.vmap(self._lstm_forecast)(x_t, h_prev, c_prev)
        c_prime = c_prime * node_mask[:, None]
        o_t = o_t * node_mask[:, None]

        # 2. Sequential Kalman Assimilation
        c_assim, kalman_trace = self.kalman(c_prime, obs_embeddings, obs_valid, obs_staleness)
        c_assim = c_assim * node_mask[:, None]

        # 3. Bidirectional Graph Message Passing
        c_spatial, spatial_trace = self.graph_mp(c_assim, static_features, tree_topology, node_mask)

        # 4. LSTM Output: h_t = o_t * tanh(c_final)
        h_new = o_t * jnp.tanh(c_spatial)
        h_new = h_new * node_mask[:, None]

        new_state = (h_new, c_spatial)
        trace = {**spatial_trace, **kalman_trace}

        return new_state, trace


# ============================================================
# Source Embedders
# ============================================================


class SourceEmbedderRegistry(eqx.Module):
    """Registry of source-specific embedders (H_s operators)."""

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


# ============================================================
# Main Model
# ============================================================


class SparseFusionGRNN_C(BaseModel):
    """
    Spatio-Temporal Graph Recurrent Neural Network with:
    - Modified LSTM with split forecast/output phases
    - Sequential Kalman-style data assimilation on the cell state
    - Bidirectional graph message passing:
        - Downstream: tree scan with static routing coefficients
        - Upstream: dilated gated messages
    - Modular source embedders (H_s operators)
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
        assim_sizes: dict[str, dict[str, int]],
        seed: int,
        dropout: float,
        head: str,
        seq2seq: bool,
        supervised_attn: bool,
        use_obs_memory: bool,
        return_weights: bool,
        num_dilations: int = 5,
        staleness_threshold: float = 14.0,
        use_sequence_vae: bool = False,
        **kwargs,
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

        # Sparse observation embedders
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

        # Spatio-temporal cell
        self.spat_temp_lstm = SpatioTemporalLSTMCell(
            lstm_input_size=hidden_size,
            hidden_size=hidden_size,
            static_feature_size=static_size,
            num_dilations=num_dilations,
            source_names=list(sparse_sizes.keys()),
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
        key = key if key is not None else jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)

        self.source_registry.register_source(
            name,
            n_features,
            embed_width=size_args.get("embed_width"),
            embed_depth=size_args.get("embed_depth", 2),
            key=k1,
        )
        self.spat_temp_lstm.kalman.register_source(name, key=k2)

    @property
    def sparse_sources(self) -> list[str]:
        return self.source_registry.sources

    def __call__(self, data: GraphBatch, key: PRNGKeyArray) -> Array:
        num_locations = data.static.shape[0]
        num_sources = len(self.source_registry.sources)
        node_mask = data.node_mask
        tree_topology = data.tree_topology

        vae_key, drop_key1, drop_key2, scan_key = jrandom.split(key, 4)

        # 0. Optional VAE
        if self.sequence_vae is not None:
            raw_dense_list = [data.dynamic[name] for name in self.dense_sources]
            raw_dense_seq = jnp.concatenate(raw_dense_list, axis=-1)
            vae_keys = jrandom.split(vae_key, num_locations)
            recon_seq, mu, logvar = jax.vmap(self.sequence_vae, in_axes=(1, 0), out_axes=(1, 0, 0))(
                raw_dense_seq, vae_keys
            )
            recon_loss = jnp.mean((raw_dense_seq - recon_seq) ** 2, axis=(0, 2))
            kl_div = -0.5 * jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=-1)
            ood_score = recon_loss + kl_div

        # 1. Raw features as scan inputs
        raw_dense = jnp.concatenate([data.dynamic[name] for name in self.dense_sources], axis=-1)
        raw_sparse = {name: data.dynamic[name] for name in self.source_registry.sources}

        # 2. Initialize state
        initial_h = jnp.zeros((num_locations, self.hidden_size))
        initial_c = jnp.zeros((num_locations, self.hidden_size))
        initial_staleness = jnp.full((num_sources, num_locations), 1e6)
        initial_state = (initial_h, initial_c, initial_staleness)

        # 3. Temporal scan
        @jax.checkpoint
        def process_one_timestep(state_prev, scan_slice, static_data, tree_topology, node_mask):
            h_prev, c_prev, staleness_prev = state_prev
            raw_dense_t, raw_sparse_t, step_key = scan_slice
            drop_key, cell_key = jrandom.split(step_key)

            # Dense embedding
            dense_emb_list = []
            for name_idx, name in enumerate(self.dense_sources):
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

            # Sparse embedding
            source_names = self.source_registry.sources
            obs_embeddings = {}
            obs_valid = {}
            obs_staleness_dict = {}
            new_staleness = staleness_prev

            for idx, name in enumerate(source_names):
                features_t = raw_sparse_t[name]
                emb_t, mask_t = self.source_registry.embed(name, features_t, static_data)

                source_staleness_new = jnp.where(mask_t, 0.0, staleness_prev[idx] + 1.0)
                new_staleness = new_staleness.at[idx].set(source_staleness_new)

                obs_embeddings[name] = emb_t
                obs_valid[name] = mask_t
                obs_staleness_dict[name] = staleness_prev[idx]

            # Cell forward
            (h_new, c_new), cell_trace = self.spat_temp_lstm(
                dense_input,
                (h_prev, c_prev),
                obs_embeddings,
                obs_valid,
                obs_staleness_dict,
                static_data,
                tree_topology,
                node_mask,
                key=cell_key,
            )

            new_state = (h_new, c_new, new_staleness)
            return new_state, h_new

        partial_timestep = partial(
            process_one_timestep,
            static_data=data.static,
            tree_topology=tree_topology,
            node_mask=node_mask,
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
