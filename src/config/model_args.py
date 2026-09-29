from typing import Literal, TypeAlias

from pydantic import BaseModel, Field


class BaseModelArgs(BaseModel):
    hidden_size: int = Field(..., gt=0)
    dropout: float = Field(..., ge=0, lt=1.0)
    head: str = "linear"
    seq2seq: bool = False
    seed: int = 0

    def as_kwargs(self) -> dict:
        return self.model_dump(exclude={"name"})


class STGATArgs(BaseModelArgs):
    name: Literal["st_gat"]
    k_hops: int = Field(..., gte=0)
    num_heads: int = Field(..., gt=0)
    return_weights: bool = False
    target: list[str] = None
    seq_length: int = None
    dense_sizes: dict[str, int] = None
    sparse_sizes: dict[str, int] = None
    static_size: int = None
    assim_sizes: dict[str, dict] = Field(default_factory=dict)
    use_obs_memory: bool = True

class STGATCArgs(BaseModelArgs):
    name: Literal["st_gat_c"]
    k_hops: int = Field(..., gte=0)
    num_heads: int = Field(..., gt=0)
    return_weights: bool = False
    target: list[str] = None
    seq_length: int = None
    dense_sizes: dict[str, int] = None
    sparse_sizes: dict[str, int] = None
    static_size: int = None
    assim_sizes: dict[str, dict] = Field(default_factory=dict)
    # seq2seq only: number of leading timesteps that produce no prediction (and no loss),
    # so the recurrent state can warm up from its zero initialization.
    spin_up: int = Field(0, ge=0)
    # Number of mixture components when head == "gmm".
    gmm_components: int = Field(100, gt=0)
    # "gru": graph messages mixed into c through a GRU update with a tanh-bounded candidate
    #   (original behavior; repeatedly squashes the cell state toward [-1, 1]).
    # "residual": c + z * delta with an unbounded delta, preserving the cell state's magnitude.
    spatial_update: Literal["gru", "residual"] = "gru"
    # "softmax": upstream messages are a convex average (original behavior).
    # "sum": each upstream edge gets an independent sigmoid gate, so tributaries add.
    fwd_aggregation: Literal["softmax", "sum"] = "softmax"
    # Concatenate the static embedding to the LSTM input at every timestep (standard for
    # multi-basin LSTMs). False: statics only reach the model through graph attention scores.
    static_to_lstm: bool = False
    # True reproduces the legacy behavior: dropout is applied to h before it is carried to
    # the next step, so a fresh mask hits the recurrent input of the LSTM gates at every
    # timestep. False: dropout is applied only to the head input.
    recurrent_dropout: bool = False


class SFGRNNArgs(BaseModelArgs):
    name: Literal["sf_grnn"]
    k_hops: int = Field(..., gte=0)
    num_heads: int = Field(..., gt=0)
    return_weights: bool = False
    supervised_attn: bool = False
    target: list[str] = None
    seq_length: int = None
    dense_sizes: dict[str, int] = None
    sparse_sizes: dict[str, int] = None
    static_size: int = None
    assim_sizes: dict[str, dict] = Field(default_factory=dict)
    use_obs_memory: bool = True
    use_sequence_vae: bool = False


class SFGRNNCArgs(BaseModelArgs):
    name: Literal["sf_grnn_c"]
    k_hops: int = Field(..., gte=0)
    num_heads: int = Field(..., gt=0)
    return_weights: bool = False
    supervised_attn: bool = False
    target: list[str] = None
    seq_length: int = None
    dense_sizes: dict[str, int] = None
    sparse_sizes: dict[str, int] = None
    static_size: int = None
    assim_sizes: dict[str, dict] = Field(default_factory=dict)
    use_obs_memory: bool = True
    use_sequence_vae: bool = False


class MCDALSTMArgs(BaseModelArgs):
    name: Literal["mc_da_lstm"]
    return_weights: bool = False
    supervised_attn: bool = False
    target: list[str] = None
    seq_length: int = None
    num_substeps: int = Field(4, gt=0)
    dense_sizes: dict[str, int] = None
    sparse_sizes: dict[str, int] = None
    static_size: int = None
    assim_sizes: dict[str, dict] = Field(default_factory=dict)
    use_obs_memory: bool = True


class LSTMGCNNArgs(BaseModelArgs):
    name: Literal["lstm_gcnn"]
    return_weights: bool = False
    supervised_attn: bool = False
    target: list[str] = None
    seq_length: int = None
    max_hops: int = Field(4, gt=0)
    dense_sizes: dict[str, int] = None
    sparse_sizes: dict[str, int] = None
    static_size: int = None
    assim_sizes: dict[str, dict] = Field(default_factory=dict)
    use_obs_memory: bool = True


class SeqAttnArgs(BaseModelArgs):
    name: Literal["lstm_mlp_attn", "flexible_hybrid"]
    num_layers: int = Field(..., gt=0)
    num_heads: int = Field(..., gt=0)
    target: list[str] = None
    seq_length: int = None
    dynamic_sizes: dict[str, int] = None
    static_size: int = None
    time_aware: dict[str, bool] = None


class StackArgs(BaseModelArgs):
    name: Literal["stacked_lstm"]
    in_targets: list[str]
    out_targets: list[str]
    dynamic_size: int = None
    static_size: int = None


class GraphLSTMArgs(BaseModelArgs):
    name: Literal["graph_lstm"]
    num_layers: int = Field(..., gt=0)
    num_heads: int = Field(..., gt=0)
    target: list[str] = None
    dynamic_size: int = None
    static_size: int = None


class MSAttnArgs(BaseModelArgs):
    name: Literal["ms_attn"]
    num_layers: int = Field(..., gt=0)
    num_heads: int = Field(..., gt=0)
    target: list[str] = None
    seq_length: int = None
    dynamic_sizes: dict[str, int] = None
    static_size: int = None
    active_source: dict[str, bool] = Field(default_factory=dict)


class RiverLSTMArgs(BaseModelArgs):
    name: Literal["river_lstm"]
    k_hops: int = Field(..., gte=0)
    return_weights: bool = False
    target: list[str] = None
    seq_length: int = None
    dense_sizes: dict[str, int] = None
    sparse_sizes: dict[str, int] = None
    static_size: int = None
    assim_sizes: dict[str, dict] = Field(default_factory=dict)
    use_obs_memory: bool = True


class RGCNArgs(BaseModelArgs):
    name: Literal["rgcn"]
    return_weights: bool = False
    target: list[str] = None
    seq_length: int = None
    dense_sizes: dict[str, int] = None
    sparse_sizes: dict[str, int] = None
    static_size: int = None
    assim_sizes: dict[str, dict] = Field(default_factory=dict)
    use_obs_memory: bool = True


class ManningArgs(BaseModelArgs):
    name: Literal["manning"]
    num_components: int = Field(..., gt=0)
    num_samples: int = Field(..., gt=0)
    depth: int = Field(..., gt=0)
    dynamic_size: int = None
    static_size: int = None
    target: str = None


ModelArgs: TypeAlias = (
    SeqAttnArgs
    | StackArgs
    | GraphLSTMArgs
    | MSAttnArgs
    | STGATArgs
    | STGATCArgs
    | SFGRNNArgs
    | SFGRNNCArgs
    | MCDALSTMArgs
    | LSTMGCNNArgs
    | RiverLSTMArgs
    | RGCNArgs
    | ManningArgs
)
